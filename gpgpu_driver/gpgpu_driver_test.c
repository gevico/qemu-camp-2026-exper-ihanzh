#include "gpgpu_driver.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DEVICE_PATH "/dev/gpgpu0"
#define VRAM_TEST_SIZE 0x10000
#define TEST_TIMEOUT_SECONDS 10

static void fatal(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "ERROR: ");
  vfprintf(stderr, fmt, ap);
  va_end(ap);
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

static void pass(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stdout, "PASS: ");
  vfprintf(stdout, fmt, ap);
  va_end(ap);
  fprintf(stdout, "\n");
}

static uint32_t read_reg32(int fd, off_t offset) {
  uint32_t value = 0;

  if (lseek(fd, offset, SEEK_SET) == (off_t)-1)
    fatal("lseek(0x%lx) failed: %s", (unsigned long)offset, strerror(errno));

  ssize_t ret = read(fd, &value, sizeof(value));
  if (ret != (ssize_t)sizeof(value))
    fatal("read reg 0x%lx failed: %s", (unsigned long)offset,
          ret < 0 ? strerror(errno) : "short read");

  return value;
}

static void write_reg32(int fd, off_t offset, uint32_t value) {
  if (lseek(fd, offset, SEEK_SET) == (off_t)-1)
    fatal("lseek(0x%lx) failed: %s", (unsigned long)offset, strerror(errno));

  ssize_t ret = write(fd, &value, sizeof(value));
  if (ret != (ssize_t)sizeof(value))
    fatal("write reg 0x%lx failed: %s", (unsigned long)offset,
          ret < 0 ? strerror(errno) : "short write");
}

static void gpgpu_ioctl_launch(int fd, uint32_t grid_x, uint32_t grid_y,
                               uint32_t grid_z, uint32_t block_x,
                               uint32_t block_y, uint32_t block_z,
                               uint32_t global_ctrl) {
  struct gpgpu_launch_params params = {
      .grid_dim = {grid_x, grid_y, grid_z},
      .block_dim = {block_x, block_y, block_z},
      .global_ctrl = global_ctrl,
  };

  int ret = ioctl(fd, GPGPU_IOCTL_LAUNCH, &params);
  if (ret < 0)
    fatal("GPGPU_IOCTL_LAUNCH failed: %s", strerror(errno));
}

static void gpgpu_test_device_id(int fd) {
  uint32_t dev_id = read_reg32(fd, GPGPU_REG_DEV_ID);
  uint32_t version = read_reg32(fd, GPGPU_REG_DEV_VERSION);

  if (dev_id != GPGPU_DEV_ID_VALUE)
    fatal("device id mismatch: got 0x%08x, expected 0x%08x", dev_id,
          GPGPU_DEV_ID_VALUE);
  if (version != GPGPU_DEV_VERSION_VALUE)
    fatal("device version mismatch: got 0x%08x, expected 0x%08x", version,
          GPGPU_DEV_VERSION_VALUE);

  pass("device identification: id=0x%08x version=0x%08x", dev_id, version);
}

static void gpgpu_test_vram_size(int fd) {
  uint32_t vram_lo = read_reg32(fd, GPGPU_REG_VRAM_SIZE_LO);
  uint32_t vram_hi = read_reg32(fd, GPGPU_REG_VRAM_SIZE_HI);
  uint64_t vram_size = ((uint64_t)vram_hi << 32) | vram_lo;

  if (vram_size != GPGPU_DEFAULT_VRAM_SIZE)
    fatal("VRAM size mismatch: got 0x%016" PRIx64 ", expected 0x%016" PRIx64,
          vram_size, (uint64_t)GPGPU_DEFAULT_VRAM_SIZE);

  pass("VRAM size registers: 0x%016llx", (unsigned long long)vram_size);
}

static void gpgpu_test_global_control(int fd) {
  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if ((status & GPGPU_STATUS_READY) != GPGPU_STATUS_READY)
    fatal("global status does not show ready (0x%08x)", status);

  write_reg32(fd, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_ENABLE);
  uint32_t ctrl = read_reg32(fd, GPGPU_REG_GLOBAL_CTRL);
  if ((ctrl & GPGPU_CTRL_ENABLE) != GPGPU_CTRL_ENABLE)
    fatal("global control write/read mismatch: got 0x%08x", ctrl);

  pass("global control registers");
}

static void gpgpu_test_dispatch_regs(int fd) {
  write_reg32(fd, GPGPU_REG_GRID_DIM_X, 64);
  write_reg32(fd, GPGPU_REG_GRID_DIM_Y, 32);
  write_reg32(fd, GPGPU_REG_GRID_DIM_Z, 1);
  write_reg32(fd, GPGPU_REG_BLOCK_DIM_X, 256);
  write_reg32(fd, GPGPU_REG_BLOCK_DIM_Y, 1);
  write_reg32(fd, GPGPU_REG_BLOCK_DIM_Z, 1);

  if (read_reg32(fd, GPGPU_REG_GRID_DIM_X) != 64 ||
      read_reg32(fd, GPGPU_REG_GRID_DIM_Y) != 32 ||
      read_reg32(fd, GPGPU_REG_GRID_DIM_Z) != 1 ||
      read_reg32(fd, GPGPU_REG_BLOCK_DIM_X) != 256 ||
      read_reg32(fd, GPGPU_REG_BLOCK_DIM_Y) != 1 ||
      read_reg32(fd, GPGPU_REG_BLOCK_DIM_Z) != 1)
    fatal("dispatch registers returned wrong values");

  pass("dispatch registers");
}

static void gpgpu_test_vram_access(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  vdata[0x0] = 0xDEADBEEF;
  vdata[0x100] = 0x12345678;
  vdata[0x1000] = 0xCAFEBABE;

  if (vdata[0x0] != 0xDEADBEEF || vdata[0x100] != 0x12345678 ||
      vdata[0x1000] != 0xCAFEBABE)
    fatal("VRAM mmap data mismatch");

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("BAR2 mmap VRAM access");
}

static void gpgpu_test_dma_registers(int fd) {
  write_reg32(fd, GPGPU_REG_DMA_SRC_LO, 0x1000);
  write_reg32(fd, GPGPU_REG_DMA_SRC_HI, 0x0);
  write_reg32(fd, GPGPU_REG_DMA_DST_LO, 0x2000);
  write_reg32(fd, GPGPU_REG_DMA_DST_HI, 0x0);
  write_reg32(fd, GPGPU_REG_DMA_SIZE, 4096);
  write_reg32(fd, GPGPU_REG_DMA_CTRL, GPGPU_DMA_START | GPGPU_DMA_DIR_TO_VRAM);

  if (read_reg32(fd, GPGPU_REG_DMA_SRC_LO) != 0x1000 ||
      read_reg32(fd, GPGPU_REG_DMA_DST_LO) != 0x2000 ||
      read_reg32(fd, GPGPU_REG_DMA_SIZE) != 4096)
    fatal("DMA registers returned wrong values");

  pass("DMA registers");
}

static void gpgpu_test_irq_regs(int fd) {

  write_reg32(fd, GPGPU_REG_IRQ_ENABLE, 0x7);
  uint32_t enable = read_reg32(fd, GPGPU_REG_IRQ_ENABLE);
  if (enable != 0x7)
    fatal("irq enable register read-back mismatch: 0x%08x", enable);

  uint32_t status = read_reg32(fd, GPGPU_REG_IRQ_STATUS);
  if (status != 0)
    fatal("irq status should be 0 after clear, got 0x%08x", status);

  pass("IRQ registers");
}

static void gpgpu_test_simt_thread_id_regs(int fd) {
  write_reg32(fd, GPGPU_REG_THREAD_ID_X, 15);
  write_reg32(fd, GPGPU_REG_THREAD_ID_Y, 7);
  write_reg32(fd, GPGPU_REG_THREAD_ID_Z, 3);

  if (read_reg32(fd, GPGPU_REG_THREAD_ID_X) != 15 ||
      read_reg32(fd, GPGPU_REG_THREAD_ID_Y) != 7 ||
      read_reg32(fd, GPGPU_REG_THREAD_ID_Z) != 3)
    fatal("SIMT thread ID registers returned wrong values");

  pass("SIMT thread ID registers");
}

static void gpgpu_test_simt_block_id_regs(int fd) {
  write_reg32(fd, GPGPU_REG_BLOCK_ID_X, 63);
  write_reg32(fd, GPGPU_REG_BLOCK_ID_Y, 31);
  write_reg32(fd, GPGPU_REG_BLOCK_ID_Z, 1);

  if (read_reg32(fd, GPGPU_REG_BLOCK_ID_X) != 63 ||
      read_reg32(fd, GPGPU_REG_BLOCK_ID_Y) != 31 ||
      read_reg32(fd, GPGPU_REG_BLOCK_ID_Z) != 1)
    fatal("SIMT block ID registers returned wrong values");

  pass("SIMT block ID registers");
}

static void gpgpu_test_simt_warp_lane_regs(int fd) {
  write_reg32(fd, GPGPU_REG_WARP_ID, 3);
  write_reg32(fd, GPGPU_REG_LANE_ID, 17);

  if (read_reg32(fd, GPGPU_REG_WARP_ID) != 3 ||
      read_reg32(fd, GPGPU_REG_LANE_ID) != 17)
    fatal("SIMT warp/lane ID registers returned wrong values");

  pass("SIMT warp/lane registers");
}

static void gpgpu_test_simt_thread_mask_reg(int fd) {
  write_reg32(fd, GPGPU_REG_THREAD_MASK, 0xFFFFFFFF);
  if (read_reg32(fd, GPGPU_REG_THREAD_MASK) != 0xFFFFFFFF)
    fatal("SIMT thread mask write/read mismatch");

  write_reg32(fd, GPGPU_REG_THREAD_MASK, 0x0000FFFF);
  if (read_reg32(fd, GPGPU_REG_THREAD_MASK) != 0x0000FFFF)
    fatal("SIMT thread mask partial write/read mismatch");

  pass("SIMT thread mask register");
}

static void gpgpu_test_simt_reset(int fd) {
  write_reg32(fd, GPGPU_REG_THREAD_ID_X, 123);
  write_reg32(fd, GPGPU_REG_BLOCK_ID_X, 456);
  write_reg32(fd, GPGPU_REG_WARP_ID, 7);
  write_reg32(fd, GPGPU_REG_THREAD_MASK, 0xDEADBEEF);

  write_reg32(fd, GPGPU_REG_GLOBAL_CTRL, GPGPU_CTRL_RESET);

  if (read_reg32(fd, GPGPU_REG_THREAD_ID_X) != 0 ||
      read_reg32(fd, GPGPU_REG_BLOCK_ID_X) != 0 ||
      read_reg32(fd, GPGPU_REG_WARP_ID) != 0 ||
      read_reg32(fd, GPGPU_REG_THREAD_MASK) != 0)
    fatal("SIMT reset did not clear registers");

  pass("SIMT reset");
}

static const uint32_t simple_kernel[] = {
    0xF1402373, /* csrrs x6, mhartid, x0 */
    0x01F37313, /* andi  x6, x6, 0x1F */
    0x00231393, /* slli  x7, x6, 2 */
    0x00001E37, /* lui   x28, 1 */
    0x007E0E33, /* add   x28, x28, x7 */
    0x006E2023, /* sw    x6, 0(x28) */
    0x00100073, /* ebreak */
};

static void gpgpu_test_kernel_exec(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  uint32_t num_threads = 8;

  for (size_t i = 0; i < sizeof(simple_kernel) / sizeof(simple_kernel[0]); i++)
    vdata[i] = simple_kernel[i];

  for (uint32_t i = 0; i < num_threads; i++)
    vdata[0x1000 / 4 + i] = 0xDEADBEEF;

  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
  gpgpu_ioctl_launch(fd, 1, 1, 1, num_threads, 1, 1, GPGPU_CTRL_ENABLE);

  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if ((status & GPGPU_STATUS_READY) != GPGPU_STATUS_READY)
    fatal("kernel exec did not complete, status=0x%08x", status);

  for (uint32_t i = 0; i < num_threads; i++) {
    uint32_t value = vdata[0x1000 / 4 + i];
    if (value != i)
      fatal("kernel exec output mismatch at %u: got %u", i, value);
  }

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("kernel exec");
}

static const uint32_t fp_kernel[] = {
    0xF1402373, /* csrrs  x6, mhartid, x0 */
    0x01F37313, /* andi   x6, x6, 0x1F */
    0xD00300D3, /* fcvt.s.w f1, x6 */
    0x00200493, /* addi   x9, x0, 2 */
    0xD0048153, /* fcvt.s.w f2, x9 */
    0x00100493, /* addi   x9, x0, 1 */
    0xD00481D3, /* fcvt.s.w f3, x9 */
    0x10208253, /* fmul.s f4, f1, f2 */
    0x003202D3, /* fadd.s f5, f4, f3 */
    0xC00293D3, /* fcvt.w.s x7, f5, RTZ */
    0x00231413, /* slli   x8, x6, 2 */
    0x00001E37, /* lui    x28, 1 */
    0x008E0E33, /* add    x28, x28, x8 */
    0x007E2023, /* sw     x7, 0(x28) */
    0x00100073, /* ebreak */
};

static void gpgpu_test_fp_kernel_exec(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  uint32_t num_threads = 8;

  for (size_t i = 0; i < sizeof(fp_kernel) / sizeof(fp_kernel[0]); i++)
    vdata[i] = fp_kernel[i];

  for (uint32_t i = 0; i < num_threads; i++)
    vdata[0x1000 / 4 + i] = 0xDEADBEEF;

  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
  gpgpu_ioctl_launch(fd, 1, 1, 1, num_threads, 1, 1, GPGPU_CTRL_ENABLE);

  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if (!(status & GPGPU_STATUS_READY))
    fatal("fp kernel exec did not complete, status=0x%08x", status);

  for (uint32_t i = 0; i < num_threads; i++) {
    uint32_t value = vdata[0x1000 / 4 + i];
    if (value != 2 * i + 1)
      fatal("fp kernel output mismatch at %u: got %u", i, value);
  }

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("FP kernel exec");
}

static const uint32_t lp_convert_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* BF16 round-trip: 42 → f32 → bf16 → f32 → int */
    0x02A00493,  /* addi   x9, x0, 42         ; x9 = 42                */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 42.0             */
    0x44108153,  /* fcvt.bf16.s f2, f1         ; f2[15:0] = bf16(42.0) */
    0x440101D3,  /* fcvt.s.bf16 f3, f2         ; f3 = f32(bf16)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = (int)f3 = 42   */

    /* E4M3 round-trip: 2 → f32 → e4m3 → f32 → int */
    0x00200493,  /* addi   x9, x0, 2          ; x9 = 2                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 2.0              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2[7:0] = e4m3(2.0)  */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3(2.0))  */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = (int)f3 = 2    */

    /* Store: output[tid*2]=x10, output[tid*2+1]=x11 */
    0x00331413,  /* slli   x8, x6, 3          ; x8 = tid * 8           */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000            */
    0x008E0E33,  /* add    x28, x28, x8       ; x28 = &output[tid*2]   */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[tid*2] = 42     */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[tid*2+1] = 2    */
    0x00100073,  /* ebreak                    ; stop                    */
};

static void gpgpu_test_lp_convert(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  uint32_t num_threads = 4;

  for (size_t i = 0;
       i < sizeof(lp_convert_kernel) / sizeof(lp_convert_kernel[0]); i++)
    vdata[i] = lp_convert_kernel[i];

  for (uint32_t i = 0; i < num_threads * 2; i++)
    vdata[0x1000 / 4 + i] = 0xDEADBEEF;

  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
  gpgpu_ioctl_launch(fd, 1, 1, 1, num_threads, 1, 1, GPGPU_CTRL_ENABLE);

  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if (!(status & GPGPU_STATUS_READY))
    fatal("lp convert kernel did not complete, status=0x%08x", status);

  for (uint32_t i = 0; i < num_threads; i++) {
    uint32_t bf16_result = vdata[0x1000 / 4 + i * 2];
    uint32_t e4m3_result = vdata[0x1000 / 4 + i * 2 + 1];
    if (bf16_result != 42 || e4m3_result != 2)
      fatal("lp convert output mismatch at %u: got %u/%u", i, bf16_result,
            e4m3_result);
  }

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("LP convert kernel");
}

static const uint32_t lp_convert_e5m2_e2m1_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* E5M2 round-trip: 4 → f32 → e5m2 → f32 → int */
    0x00400493,  /* addi   x9, x0, 4          ; x9 = 4                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 4.0              */
    0x48308153,  /* fcvt.e5m2.s f2, f1         ; f2 = e5m2(4.0)        */
    0x482101D3,  /* fcvt.s.e5m2 f3, f2         ; f3 = f32(e5m2)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = 4               */

    /* E2M1 round-trip: 2 → f32 → e2m1 → f32 → int */
    0x00200493,  /* addi   x9, x0, 2          ; x9 = 2                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 2.0              */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(2.0)        */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = f32(e2m1)        */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = 2               */

    /* BF16 round-trip: -3 → f32 → bf16 → f32 → int */
    0xFFD00493,  /* addi   x9, x0, -3         ; x9 = -3                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = -3.0             */
    0x44108153,  /* fcvt.bf16.s f2, f1         ; f2 = bf16(-3.0)        */
    0x440101D3,  /* fcvt.s.bf16 f3, f2         ; f3 = f32(bf16)         */
    0xC0019653,  /* fcvt.w.s x12, f3, rtz      ; x12 = -3              */

    /* E4M3 round-trip: -2 → f32 → e4m3 → f32 → int */
    0xFFE00493,  /* addi   x9, x0, -2         ; x9 = -2                 */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = -2.0             */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(-2.0)       */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3)        */
    0xC00196D3,  /* fcvt.w.s x13, f3, rtz      ; x13 = -2              */

    /* Store 4 results: tid * 16 offset */
    0x00431413,  /* slli   x8, x6, 4          ; x8 = tid * 16           */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000             */
    0x008E0E33,  /* add    x28, x28, x8       ; x28 = base + offset     */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[0] = 4           */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[1] = 2           */
    0x00CE2423,  /* sw     x12, 8(x28)        ; output[2] = -3          */
    0x00DE2623,  /* sw     x13, 12(x28)       ; output[3] = -2          */
    0x00100073,  /* ebreak                    ; stop                     */
};

static void gpgpu_test_lp_convert_e5m2_e2m1(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  int32_t expected[] = {4, 2, -3, -2};

  for (size_t i = 0; i < sizeof(lp_convert_e5m2_e2m1_kernel) /
                             sizeof(lp_convert_e5m2_e2m1_kernel[0]);
       i++)
    vdata[i] = lp_convert_e5m2_e2m1_kernel[i];

  for (int i = 0; i < 4; i++)
    vdata[0x1000 / 4 + i] = 0xDEADBEEF;

  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
  gpgpu_ioctl_launch(fd, 1, 1, 1, 1, 1, 1, GPGPU_CTRL_ENABLE);

  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if (!(status & GPGPU_STATUS_READY))
    fatal("LP E5M2/E2M1 kernel did not complete, status=0x%08x", status);

  for (int i = 0; i < 4; i++) {
    int32_t value = (int32_t)vdata[0x1000 / 4 + i];
    if (value != expected[i])
      fatal("LP E5M2/E2M1 kernel output mismatch at %d: got %d", i, value);
  }

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("LP E5M2/E2M1 kernel");
}

static const uint32_t lp_convert_saturate_kernel[] = {
    0xF1402373,  /* csrrs  x6, mhartid, x0    ; x6 = mhartid           */
    0x01F37313,  /* andi   x6, x6, 0x1F       ; x6 = tid               */

    /* E4M3 round-trip: 0 → f32 → e4m3 → f32 → int */
    0x00000493,  /* addi   x9, x0, 0          ; x9 = 0                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 0.0              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(0.0)        */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = f32(e4m3)        */
    0xC0019553,  /* fcvt.w.s x10, f3, rtz      ; x10 = 0               */

    /* E2M1 round-trip: 0 → f32 → e2m1 → f32 → int */
    0x00000493,  /* addi   x9, x0, 0          ; x9 = 0                  */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 0.0              */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(0.0)        */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = f32(e2m1)        */
    0xC00195D3,  /* fcvt.w.s x11, f3, rtz      ; x11 = 0               */

    /* E2M1 round-trip: 100 → saturate → 6 */
    0x06400493,  /* addi   x9, x0, 100        ; x9 = 100                */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 100.0            */
    0x4C108153,  /* fcvt.e2m1.s f2, f1         ; f2 = e2m1(100.0) sat  */
    0x4C0101D3,  /* fcvt.s.e2m1 f3, f2         ; f3 = 6.0              */
    0xC0019653,  /* fcvt.w.s x12, f3, rtz      ; x12 = 6               */

    /* E4M3 round-trip: 1000 → saturate → 448 */
    0x3E800493,  /* addi   x9, x0, 1000       ; x9 = 1000              */
    0xD00480D3,  /* fcvt.s.w  f1, x9           ; f1 = 1000.0           */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(1000) sat   */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = 448.0            */
    0xC00196D3,  /* fcvt.w.s x13, f3, rtz      ; x13 = 448             */

    /* E4M3 of +Inf → saturate → 448 */
    0x7F8004B7,  /* lui    x9, 0x7F800        ; x9 = 0x7F800000 (+Inf) */
    0xF00480D3,  /* fmv.w.x f1, x9            ; f1 = +Inf              */
    0x48108153,  /* fcvt.e4m3.s f2, f1         ; f2 = e4m3(Inf) sat    */
    0x480101D3,  /* fcvt.s.e4m3 f3, f2         ; f3 = 448.0            */
    0xC0019753,  /* fcvt.w.s x14, f3, rtz      ; x14 = 448             */

    /* Store 5 results: tid * 20 offset (tid=0 → base=0x1000) */
    0x00001E37,  /* lui    x28, 1             ; x28 = 0x1000             */
    0x00AE2023,  /* sw     x10, 0(x28)        ; output[0] = 0           */
    0x00BE2223,  /* sw     x11, 4(x28)        ; output[1] = 0           */
    0x00CE2423,  /* sw     x12, 8(x28)        ; output[2] = 6           */
    0x00DE2623,  /* sw     x13, 12(x28)       ; output[3] = 448         */
    0x00EE2823,  /* sw     x14, 16(x28)       ; output[4] = 448         */
    0x00100073,  /* ebreak                    ; stop                     */
};

static void gpgpu_test_lp_convert_saturate(int fd) {
  void *vram =
      mmap(NULL, VRAM_TEST_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (vram == MAP_FAILED)
    fatal("mmap BAR2 failed: %s", strerror(errno));

  volatile uint32_t *vdata = vram;
  int32_t expected[] = {0, 0, 6, 448, 448};

  for (size_t i = 0; i < sizeof(lp_convert_saturate_kernel) /
                             sizeof(lp_convert_saturate_kernel[0]);
       i++)
    vdata[i] = lp_convert_saturate_kernel[i];

  for (int i = 0; i < 5; i++)
    vdata[0x1000 / 4 + i] = 0xDEADBEEF;

  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_LO, 0x0000);
  write_reg32(fd, GPGPU_REG_KERNEL_ADDR_HI, 0x0000);
  gpgpu_ioctl_launch(fd, 1, 1, 1, 1, 1, 1, GPGPU_CTRL_ENABLE);

  uint32_t status = read_reg32(fd, GPGPU_REG_GLOBAL_STATUS);
  if (!(status & GPGPU_STATUS_READY))
    fatal("LP saturate kernel did not complete, status=0x%08x", status);

  for (int i = 0; i < 5; i++) {
    int32_t value = (int32_t)vdata[0x1000 / 4 + i];
    if (value != expected[i])
      fatal("LP saturate output mismatch at %d: got %d", i, value);
  }

  if (munmap((void *)vram, VRAM_TEST_SIZE) != 0)
    fatal("munmap failed: %s", strerror(errno));

  pass("LP saturate kernel");
}

static void run_test(int fd, const char *name, void (*func)(int)) {
  fprintf(stdout, "[RUN] %s\n", name);
  func(fd);
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : DEVICE_PATH;
  int fd = open(path, O_RDWR);
  if (fd < 0)
    fatal("open('%s') failed: %s", path, strerror(errno));

  run_test(fd, "device identification", gpgpu_test_device_id);
  run_test(fd, "VRAM size", gpgpu_test_vram_size);
  run_test(fd, "global control", gpgpu_test_global_control);
  run_test(fd, "dispatch registers", gpgpu_test_dispatch_regs);
  run_test(fd, "BAR2 mmap VRAM access", gpgpu_test_vram_access);
  run_test(fd, "DMA registers", gpgpu_test_dma_registers);
  run_test(fd, "IRQ registers", gpgpu_test_irq_regs);
  run_test(fd, "SIMT thread ID registers", gpgpu_test_simt_thread_id_regs);
  run_test(fd, "SIMT block ID registers", gpgpu_test_simt_block_id_regs);
  run_test(fd, "SIMT warp/lane registers", gpgpu_test_simt_warp_lane_regs);
  run_test(fd, "SIMT thread mask register", gpgpu_test_simt_thread_mask_reg);
  run_test(fd, "SIMT reset", gpgpu_test_simt_reset);
  run_test(fd, "kernel exec", gpgpu_test_kernel_exec);
  run_test(fd, "FP kernel exec", gpgpu_test_fp_kernel_exec);
  run_test(fd, "LP convert kernel", gpgpu_test_lp_convert);
  run_test(fd, "LP E5M2/E2M1 kernel", gpgpu_test_lp_convert_e5m2_e2m1);
  run_test(fd, "LP saturate kernel", gpgpu_test_lp_convert_saturate);

  close(fd);
  printf("All tests passed.\n");
  return EXIT_SUCCESS;
}
