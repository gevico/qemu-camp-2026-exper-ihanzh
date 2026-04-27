#ifndef GPGPU_DRIVER_H
#define GPGPU_DRIVER_H

#include <linux/ioctl.h>
#include <linux/types.h>

/* 设备标识值 */
#define GPGPU_REVISION 0x01                /* 硬件版本号 */
#define GPGPU_DEV_ID_VALUE 0x47505055      /* "GPPU" in ASCII */
#define GPGPU_DEV_VERSION_VALUE 0x00010000 /* v1.0.0 */

#define GPGPU_IOCTL_MAGIC 'G'
#define GPGPU_IOCTL_LAUNCH                                                     \
  _IOW(GPGPU_IOCTL_MAGIC, 0, struct gpgpu_launch_params)

/* PCI 配置 */
#define GPGPU_VENDOR_ID 0x1234
#define GPGPU_DEVICE_ID 0x1337

/* 设备信息寄存器 (BAR0) */
#define GPGPU_REG_DEV_ID 0x0000
#define GPGPU_REG_DEV_VERSION 0x0004
#define GPGPU_REG_DEV_CAPS 0x0008
#define GPGPU_REG_VRAM_SIZE_LO 0x000C
#define GPGPU_REG_VRAM_SIZE_HI 0x0010

/* 全局控制寄存器 */
#define GPGPU_REG_GLOBAL_CTRL 0x0100
#define GPGPU_REG_GLOBAL_STATUS 0x0104
#define GPGPU_REG_ERROR_STATUS 0x0108

/* 中断寄存器 */
#define GPGPU_REG_IRQ_ENABLE 0x0200
#define GPGPU_REG_IRQ_STATUS 0x0204
#define GPGPU_REG_IRQ_ACK 0x0208

/* 内核分发寄存器 */
#define GPGPU_REG_GRID_DIM_X 0x0310
#define GPGPU_REG_GRID_DIM_Y 0x0314
#define GPGPU_REG_GRID_DIM_Z 0x0318
#define GPGPU_REG_BLOCK_DIM_X 0x031C
#define GPGPU_REG_BLOCK_DIM_Y 0x0320
#define GPGPU_REG_BLOCK_DIM_Z 0x0324

/* DMA 寄存器 */
#define GPGPU_REG_DMA_SRC_LO 0x0400
#define GPGPU_REG_DMA_SRC_HI 0x0404
#define GPGPU_REG_DMA_DST_LO 0x0408
#define GPGPU_REG_DMA_DST_HI 0x040C
#define GPGPU_REG_DMA_SIZE 0x0410
#define GPGPU_REG_DMA_CTRL 0x0414
#define GPGPU_REG_DMA_STATUS 0x0418

/* SIMT 上下文寄存器 (CTRL 设备) */
#define GPGPU_REG_THREAD_ID_X 0x1000
#define GPGPU_REG_THREAD_ID_Y 0x1004
#define GPGPU_REG_THREAD_ID_Z 0x1008
#define GPGPU_REG_BLOCK_ID_X 0x1010
#define GPGPU_REG_BLOCK_ID_Y 0x1014
#define GPGPU_REG_BLOCK_ID_Z 0x1018
#define GPGPU_REG_WARP_ID 0x1020
#define GPGPU_REG_LANE_ID 0x1024

/* 同步寄存器 */
#define GPGPU_REG_BARRIER 0x2000
#define GPGPU_REG_THREAD_MASK 0x2004

/* 内核地址寄存器 */
#define GPGPU_REG_KERNEL_ADDR_LO 0x0300
#define GPGPU_REG_KERNEL_ADDR_HI 0x0304
#define GPGPU_REG_DISPATCH 0x0330

/* GLOBAL_CTRL 寄存器位 */
#define GPGPU_CTRL_ENABLE (1 << 0) /* 设备使能位 */
#define GPGPU_CTRL_RESET (1 << 1)  /* 软复位 (自动清除) */

/* GLOBAL_STATUS 寄存器位 */
#define GPGPU_STATUS_READY (1 << 0) /* 设备就绪 */
#define GPGPU_STATUS_BUSY (1 << 1)  /* 设备忙 (内核执行中) */
#define GPGPU_STATUS_ERROR (1 << 2) /* 发生错误 */

/* ERROR_STATUS 寄存器位 (写 1 清除) */
#define GPGPU_ERR_INVALID_CMD (1 << 0)  /* 无效命令 */
#define GPGPU_ERR_VRAM_FAULT (1 << 1)   /* 显存访问越界 */
#define GPGPU_ERR_KERNEL_FAULT (1 << 2) /* 内核执行错误 */
#define GPGPU_ERR_DMA_FAULT (1 << 3)    /* DMA 传输错误 */

/* IRQ 位定义 */
#define GPGPU_IRQ_KERNEL_DONE (1 << 0) /* 内核执行完成中断 */
#define GPGPU_IRQ_DMA_DONE (1 << 1)    /* DMA 传输完成中断 */
#define GPGPU_IRQ_ERROR (1 << 2)       /* 错误中断 */

/* DMA_CTRL 寄存器位 */
#define GPGPU_DMA_START (1 << 0)         /* 启动 DMA 传输 */
#define GPGPU_DMA_DIR_TO_VRAM (0 << 1)   /* 方向: 主机 -> 显存 */
#define GPGPU_DMA_DIR_FROM_VRAM (1 << 1) /* 方向: 显存 -> 主机 */
#define GPGPU_DMA_IRQ_ENABLE (1 << 2)    /* 完成时产生中断 */

/* DMA_STATUS 寄存器位 */
#define GPGPU_DMA_BUSY (1 << 0)     /* DMA 忙 */
#define GPGPU_DMA_COMPLETE (1 << 1) /* DMA 完成 */
#define GPGPU_DMA_ERROR (1 << 2)    /* DMA 错误 */

/* 默认配置 */
#define GPGPU_DEFAULT_VRAM_SIZE (64 * 1024 * 1024)

struct gpgpu_launch_params {
  __u32 grid_dim[3];
  __u32 block_dim[3];
  __u32 global_ctrl;
};

#endif // GPGPU_DRIVER_H
