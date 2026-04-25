/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include <math.h>
#ifndef GPGPU_CORE_USE_ACCESSORS
#include "gpgpu.h"
#endif
#include "gpgpu_core.h"

#ifdef GPGPU_CORE_USE_ACCESSORS
static inline uint64_t gpgpu_core_vram_size(GPGPUState *s)
{
    return gpgpu_get_vram_size(s);
}

static inline uint8_t *gpgpu_core_vram_ptr(GPGPUState *s)
{
    return gpgpu_get_vram_ptr(s);
}

static inline GPGPUKernelParams gpgpu_core_kernel(GPGPUState *s)
{
    return gpgpu_get_kernel(s);
}
#else
static inline uint64_t gpgpu_core_vram_size(GPGPUState *s)
{
    return s->vram_size;
}

static inline uint8_t *gpgpu_core_vram_ptr(GPGPUState *s)
{
    return s->vram_ptr;
}

static inline GPGPUKernelParams gpgpu_core_kernel(GPGPUState *s)
{
    return s->kernel;
}
#endif

/* TODO: Implement warp initialization */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc, uint32_t thread_id_base,
                          const uint32_t block_id[3], uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    memset(warp, 0, sizeof(*warp));
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];
    warp->warp_id = warp_id;
    warp->thread_id_base = thread_id_base;
    for (uint32_t i = 0; i < num_threads; i++) {
        warp->lanes[i].pc = pc;
        /* 线程 ID 由 block_id, warp_id 和 lane_id 计算得出 */
        uint32_t thread_id = (block_id_linear * (GPGPU_DEFAULT_WARP_SIZE *
                                                 GPGPU_DEFAULT_WARPS_PER_CU)) +
                             (warp_id * GPGPU_DEFAULT_WARP_SIZE) + i;
        warp->lanes[i].gpr[0] = thread_id; /* x0 寄存器保存线程 ID */
        warp->lanes[i].mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        warp->lanes[i].active = true;
    }
    warp->active_mask =
        (1u << num_threads) - 1; /* 前 num_threads 个 lane 活跃 */
}

#define R(i) lane->gpr[i]
#define F(i) lane->fpr[i]
#define Mr vmem_read
#define Mw vmem_write

#ifndef BITS
#define BITS(val, hi, lo) (((val) >> (lo)) & (~0u >> (31 - (hi) + (lo))))
#endif
#ifndef SEXT
#define SEXT(val, bits) ((int32_t)((val) << (32 - (bits))) >> (32 - (bits)))
#endif

static inline uint32_t vmem_read(GPGPUState *s, uint32_t addr, int size)
{
    uint64_t vram_size = gpgpu_core_vram_size(s);
    const uint8_t *vram_ptr = gpgpu_core_vram_ptr(s);

    g_assert(vram_ptr != NULL);
    g_assert((uint64_t)addr + size <= vram_size);
    const uint8_t *ptr = vram_ptr + addr;
    switch (size) {
    case 1:
        return ptr[0];
    case 2:
        return ptr[0] | (ptr[1] << 8);
    case 4:
        return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    default:
        g_assert_not_reached();
    }
}

static inline void vmem_write(GPGPUState *s, uint32_t addr, int size,
                              uint32_t value)
{
    uint64_t vram_size = gpgpu_core_vram_size(s);
    uint8_t *vram_ptr = gpgpu_core_vram_ptr(s);

    g_assert(vram_ptr != NULL);
    g_assert((uint64_t)addr + size <= vram_size);
    uint8_t *ptr = vram_ptr + addr;
    switch (size) {
    case 1:
        ptr[0] = value & 0xff;
        break;
    case 2:
        ptr[0] = value & 0xff;
        ptr[1] = (value >> 8) & 0xff;
        break;
    case 4:
        ptr[0] = value & 0xff;
        ptr[1] = (value >> 8) & 0xff;
        ptr[2] = (value >> 16) & 0xff;
        ptr[3] = (value >> 24) & 0xff;
        break;
    default:
        g_assert_not_reached();
    }
}

static inline uint32_t extract_bits(uint32_t value, int hi, int lo)
{
    return (value >> lo) & (~0u >> (31 - hi + lo));
}

static inline int32_t sign_extend(uint32_t value, int bits)
{
    uint32_t shift = 32 - bits;
    return (int32_t)(value << shift) >> shift;
}

static inline float uint32_to_float(uint32_t value)
{
    union {
        uint32_t u;
        float f;
    } conv = { .u = value };
    return conv.f;
}

static inline uint32_t float_to_uint32(float value)
{
    union {
        float f;
        uint32_t u;
    } conv = { .f = value };
    return conv.u;
}

static inline float4_e2m1 float32_to_float4_e2m1(uint32_t value,
                                                 float_status *status)
{
    float f = uint32_to_float(value);
    if (isnan(f)) {
        return (float4_e2m1)0x7u; /* quiet NaN not fully defined here */
    }

    bool neg = signbit(f);
    float af = fabsf(f);
    uint32_t bits;

    if (af < 0.25f) {
        bits = 0x0u; /* zero */
    } else if (af < 0.75f) {
        bits = 0x1u; /* 0.5 */
    } else if (af < 1.25f) {
        bits = 0x2u; /* 1.0 */
    } else if (af < 1.75f) {
        bits = 0x3u; /* 1.5 */
    } else if (af < 2.5f) {
        bits = 0x4u; /* 2.0 */
    } else if (af < 3.5f) {
        bits = 0x5u; /* 3.0 */
    } else if (af < 5.0f) {
        bits = 0x6u; /* 4.0 */
    } else {
        bits = 0x7u; /* 6.0 saturate */
    }

    return (float4_e2m1)(bits | (neg ? 0x8u : 0));
}

static inline int32_t imm_i(uint32_t inst)
{
    return sign_extend(inst >> 20, 12);
}

static inline uint32_t imm_u(uint32_t inst)
{
    return inst & 0xfffff000u;
}

static inline int32_t imm_s(uint32_t inst)
{
    uint32_t imm = ((inst >> 25) << 5) | ((inst >> 7) & 0x1f);
    return sign_extend(imm, 12);
}

static inline int32_t imm_b(uint32_t inst)
{
    uint32_t imm = ((inst >> 31) & 1) << 12;
    imm |= ((inst >> 7) & 1) << 11;
    imm |= ((inst >> 25) & 0x3f) << 5;
    imm |= ((inst >> 8) & 0xf) << 1;
    return sign_extend(imm, 13);
}

static inline int32_t imm_j(uint32_t inst)
{
    uint32_t imm = ((inst >> 31) & 1) << 20;
    imm |= ((inst >> 21) & 0x3ff) << 1;
    imm |= ((inst >> 20) & 1) << 11;
    imm |= ((inst >> 12) & 0xff) << 12;
    return sign_extend(imm, 21);
}

static void decode(GPGPUState *s, GPGPULane *lane, uint32_t inst)
{
    uint32_t opcode = inst & 0x7f;
    uint32_t rd = extract_bits(inst, 11, 7);
    uint32_t funct3 = extract_bits(inst, 14, 12);
    // uint32_t rm = extract_bits(inst, 14, 12);
    uint32_t rs1 = extract_bits(inst, 19, 15);
    uint32_t rs2 = extract_bits(inst, 24, 20);
    uint32_t rs3 = extract_bits(inst, 31, 27);
    uint32_t funct7 = extract_bits(inst, 31, 25);
    uint32_t next_pc = lane->pc + 4;
    uint32_t src1 = lane->gpr[rs1];
    uint32_t src2 = lane->gpr[rs2];
    uint32_t fsrc1_bits = lane->fpr[rs1];
    uint32_t fsrc2_bits = lane->fpr[rs2];
    uint32_t fsrc3_bits = lane->fpr[rs3];
    float fsrc1 = uint32_to_float(fsrc1_bits);
    float fsrc2 = uint32_to_float(fsrc2_bits);
    float fsrc3 = uint32_to_float(fsrc3_bits);
    const char *opname = "unknown";
    uint32_t op_imm = 0;
    uint32_t result = 0;
    bool is_store = false;
    bool flag = false; /* 是否打印指令执行日志，调试时可设置为 true */

    switch (opcode) {
    case 0x37: /* LUI */
        opname = "LUI";
        op_imm = imm_u(inst);
        R(rd) = op_imm;
        break;
    case 0x17: /* AUIPC */
        opname = "AUIPC";
        op_imm = imm_u(inst);
        R(rd) = lane->pc + op_imm;
        break;
    case 0x6f: /* JAL */
        opname = "JAL";
        op_imm = imm_j(inst);
        R(rd) = lane->pc + 4;
        next_pc = lane->pc + op_imm;
        break;
    case 0x67: /* JALR */
        opname = "JALR";
        op_imm = imm_i(inst);
        if (funct3 == 0) {
            R(rd) = lane->pc + 4;
            next_pc = (src1 + op_imm) & ~1u;
        }
        break;
    case 0x63: /* B-type */
        op_imm = imm_b(inst);
        switch (funct3) {
        case 0:
            opname = "BEQ";
            if (src1 == src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        case 1:
            opname = "BNE";
            if (src1 != src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        case 4:
            opname = "BLT";
            if ((int32_t)src1 < (int32_t)src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        case 5:
            opname = "BGE";
            if ((int32_t)src1 >= (int32_t)src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        case 6:
            opname = "BLTU";
            if (src1 < src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        case 7:
            opname = "BGEU";
            if (src1 >= src2) {
                next_pc = lane->pc + op_imm;
            }
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case 0x03: /* Loads */
        op_imm = imm_i(inst);
        switch (funct3) {
        case 0:
            opname = "LB";
            R(rd) = (int32_t)(int8_t)Mr(s, src1 + op_imm, 1);
            break;
        case 1:
            opname = "LH";
            R(rd) = (int32_t)(int16_t)Mr(s, src1 + op_imm, 2);
            break;
        case 2:
            opname = "LW";
            R(rd) = Mr(s, src1 + op_imm, 4);
            break;
        case 4:
            opname = "LBU";
            R(rd) = Mr(s, src1 + op_imm, 1);
            break;
        case 5:
            opname = "LHU";
            R(rd) = Mr(s, src1 + op_imm, 2);
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case 0x23: /* Stores */
        op_imm = imm_s(inst);
        is_store = true;
        switch (funct3) {
        case 0:
            opname = "SB";
            Mw(s, src1 + op_imm, 1, src2);
            break;
        case 1:
            opname = "SH";
            Mw(s, src1 + op_imm, 2, src2);
            break;
        case 2:
            opname = "SW";
            Mw(s, src1 + op_imm, 4, src2);
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case 0x13: /* Immediate arithmetic */
        op_imm = imm_i(inst);
        switch (funct3) {
        case 0:
            opname = "ADDI";
            R(rd) = src1 + op_imm;
            break;
        case 2:
            opname = "SLTI";
            R(rd) = ((int32_t)src1 < op_imm) ? 1 : 0;
            break;
        case 3:
            opname = "SLTIU";
            R(rd) = (src1 < (uint32_t)op_imm) ? 1 : 0;
            break;
        case 4:
            opname = "XORI";
            R(rd) = src1 ^ op_imm;
            break;
        case 6:
            opname = "ORI";
            R(rd) = src1 | op_imm;
            break;
        case 7:
            opname = "ANDI";
            R(rd) = src1 & op_imm;
            break;
        case 1:
            opname = "SLLI";
            if (funct7 == 0x00) {
                R(rd) = src1 << (op_imm & 0x1f);
            } else {
                g_assert_not_reached();
            }
            break;
        case 5:
            opname = (funct7 == 0x00) ? "SRLI" : "SRAI";
            if (funct7 == 0x00) {
                R(rd) = src1 >> (op_imm & 0x1f);
            } else if (funct7 == 0x20) {
                R(rd) = (int32_t)src1 >> (op_imm & 0x1f);
            } else {
                g_assert_not_reached();
            }
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case 0x33: /* Register arithmetic */
        switch (funct3) {
        case 0:
            opname = (funct7 == 0x00) ? "ADD" : "SUB";
            if (funct7 == 0x00) {
                R(rd) = src1 + src2;
            } else if (funct7 == 0x20) {
                R(rd) = src1 - src2;
            } else {
                g_assert_not_reached();
            }
            break;
        case 1:
            opname = "SLL";
            R(rd) = src1 << (src2 & 0x1f);
            break;
        case 2:
            opname = "SLT";
            R(rd) = ((int32_t)src1 < (int32_t)src2) ? 1 : 0;
            break;
        case 3:
            opname = "SLTU";
            R(rd) = (src1 < src2) ? 1 : 0;
            break;
        case 4:
            opname = "XOR";
            R(rd) = src1 ^ src2;
            break;
        case 5:
            opname = (funct7 == 0x00) ? "SRL" : "SRA";
            if (funct7 == 0x00) {
                R(rd) = src1 >> (src2 & 0x1f);
            } else if (funct7 == 0x20) {
                R(rd) = (int32_t)src1 >> (src2 & 0x1f);
            } else {
                g_assert_not_reached();
            }
            break;
        case 6:
            opname = "OR";
            R(rd) = src1 | src2;
            break;
        case 7:
            opname = "AND";
            R(rd) = src1 & src2;
            break;
        default:
            g_assert_not_reached();
        }
        break;
    case 0x73: /* SYSTEM */
        if (inst == 0x00100073u) {
            opname = "EBREAK";
            lane->active = false;
            lane->gpr[0] = 0;
            if (flag) {
                fprintf(stderr,
                        "GPGPU decode lane=%u pc=0x%08x inst=0x%08x %s\n",
                        lane->mhartid & MHARTID_THREAD_MASK, lane->pc, inst,
                        opname);
            }
            return;
        }
        if (funct3 == 2) { /* CSRRS */
            uint32_t csr = inst >> 20;
            opname = "CSRRS";
            if (csr == CSR_MHARTID) {
                if (rd != 0) {
                    R(rd) = lane->mhartid;
                }
            } else {
                g_assert_not_reached();
            }
        } else {
            g_assert_not_reached();
        }
        break;
    case 0x53:
        switch (funct7) {
        case 0x00:
            opname = "FADD.S";
            F(rd) = float_to_uint32(fsrc1 + fsrc2);
            break;
        case 0x04:
            opname = "FSUB.S";
            F(rd) = float_to_uint32(fsrc1 - fsrc2);
            break;
        case 0x08:
            opname = "FMUL.S";
            F(rd) = float_to_uint32(fsrc1 * fsrc2);
            break;
        case 0x0c:
            opname = "FDIV.S";
            F(rd) = float_to_uint32(fsrc1 / fsrc2);
            break;
        case 0x2c:
            if (rs2 == 0) {
                opname = "FSORT.S"; // computes the square root of rs1
                F(rd) = float_to_uint32(sqrtf(fsrc1));
                break;
            }
            break;
        case 0x10:
            if (funct3 == 0x0) {
                opname = "FSGNJ.S";
                F(rd) = (fsrc1_bits & 0x7fffffff) | (fsrc2_bits & 0x80000000u);
            } else if (funct3 == 0x1) {
                opname = "FSGNJN.S";
                F(rd) = (fsrc1_bits & 0x7fffffff) | (~fsrc2_bits & 0x80000000u);
            } else if (funct3 == 0x2) {
                opname = "FSGNJX.S";
                F(rd) = fsrc1_bits ^ (fsrc2_bits & 0x80000000u);
            }
            break;


        case 0x14:
            if (funct3 == 0x0) {
                opname = "FMIN.S";
                F(rd) = float_to_uint32((fsrc1 < fsrc2) ? fsrc1 : fsrc2);
            } else if (funct3 == 0x1) {
                opname = "FMAX.S";
                F(rd) = float_to_uint32((fsrc1 > fsrc2) ? fsrc1 : fsrc2);
            }
            break;

        case 0x50: {
            if (funct3 == 0x2) {
                opname = "FEQ.S";
                F(rd) = float_to_uint32((fsrc1 == fsrc2) ? 1.0f : 0.0f);
            } else if (funct3 == 0x1) {
                opname = "FLT.S";
                F(rd) = float_to_uint32((fsrc1 < fsrc2) ? 1.0f : 0.0f);
            } else if (funct3 == 0x0) {
                opname = "FLE.S";
                F(rd) = float_to_uint32((fsrc1 <= fsrc2) ? 1.0f : 0.0f);
            }
            break;
        }
        case 0x60:
            if (rs2 == 0x0) {
                opname = "FCVT.W.S";
                R(rd) = (int32_t)fsrc1;
            } else if (rs2 == 0x1) {
                opname = "FCVT.WU.S";
                R(rd) = (uint32_t)fsrc1;
            }
            break;

        case 0x68:
            if (rs2 == 0x0) {
                opname = "FCVT.S.W";
                F(rd) = float_to_uint32((float)(int32_t)src1);
            } else if (rs2 == 0x1) {
                opname = "FCVT.S.WU";
                F(rd) = float_to_uint32((float)(uint32_t)src1);
            }
            break;

        case 0x70:
            if (rs2 == 0x0 && funct3 == 0x0) {
                opname = "FMV.X.W";
                R(rd) = fsrc1_bits;
            } else if (rs2 == 0x0 && funct3 == 0x1) {
                opname = "FMV.W.X";
                F(rd) = src1;
            }
            break;

        case 0x78:
            if (rs2 == 0x0 && funct3 == 0x0) {
                opname = "FMV.W.X";
                F(rd) = src1;
            }
            break;

        case 0x22:
            if (rs2 == 0x0) {
                opname = "FCVT.S.BF16"; // BF16->FP32
                {
                    bfloat16 bf = (bfloat16)(fsrc1_bits & 0xffffu);
                    F(rd) = bfloat16_to_float32(bf, &lane->fp_status);
                }
            } else if (rs2 == 0x1) {
                opname = "FCVT.BF16.S"; // FP32->BF16
                {
                    bfloat16 bf =
                        float32_to_bfloat16(fsrc1_bits, &lane->fp_status);
                    F(rd) = (uint32_t)bf;
                }
            }
            break;

        case 0x24:
            if (rs2 == 0x0) {
                opname = "FCVT.S.E4M3"; // E4M3 → FP32（经 BF16 中转）
                {
                    float8_e4m3 e = (float8_e4m3)(fsrc1_bits & 0xffu);
                    bfloat16 bf = float8_e4m3_to_bfloat16(e, &lane->fp_status);
                    F(rd) = bfloat16_to_float32(bf, &lane->fp_status);
                }
            } else if (rs2 == 0x1) {
                opname = "FCVT.E4M3.S"; // FP32 → E4M3（饱和模式）
                {
                    float8_e4m3 e = float32_to_float8_e4m3(fsrc1_bits, true,
                                                           &lane->fp_status);
                    F(rd) = (uint32_t)e;
                }
            } else if (rs2 == 0x2) {
                opname = "FCVT.S.E5M2"; // E5M2 → FP32（经 BF16 中转）
                {
                    float8_e5m2 e = (float8_e5m2)(fsrc1_bits & 0xffu);
                    bfloat16 bf = float8_e5m2_to_bfloat16(e, &lane->fp_status);
                    F(rd) = bfloat16_to_float32(bf, &lane->fp_status);
                }
            } else if (rs2 == 0x3) {
                opname = "FCVT.E5M2.S"; // FP32 → E5M2（饱和模式）
                {
                    float8_e5m2 e = float32_to_float8_e5m2(fsrc1_bits, true,
                                                           &lane->fp_status);
                    F(rd) = (uint32_t)e;
                }
            }
            break;

        case 0x26:
            if (rs2 == 0x0) {
                opname =
                    "FCVT.S.E2M1"; // E2M1 → FP32（链：E2M1→E4M3→BF16→FP32）
                {
                    float4_e2m1 e = (float4_e2m1)(fsrc1_bits & 0xfu);
                    float8_e4m3 e4 =
                        float4_e2m1_to_float8_e4m3(e, &lane->fp_status);
                    bfloat16 bf = float8_e4m3_to_bfloat16(e4, &lane->fp_status);
                    F(rd) = bfloat16_to_float32(bf, &lane->fp_status);
                }
            } else if (rs2 == 0x1) {
                opname =
                    "FCVT.E2M1.S"; // FP32 → E2M1（手写阈值舍入，饱和到 ±6.0）
                F(rd) = (uint32_t)float32_to_float4_e2m1(fsrc1_bits,
                                                         &lane->fp_status);
            }
            break;

        default:
            break;
        }
        break;

    case 0x43:
        opname = "FMADD.S";
        F(rd) = float_to_uint32(fsrc1 * fsrc2 + fsrc3);
        break;

    case 0x47:
        opname = "FMSUB.S";
        F(rd) = float_to_uint32(fsrc1 * fsrc2 - fsrc3);
        break;

    case 0x4b:
        opname = "FNMSUB.S";
        F(rd) = float_to_uint32(-(fsrc1 * fsrc2) + fsrc3);
        break;

    case 0x4f:
        opname = "FNMADD.S";
        F(rd) = float_to_uint32(-(fsrc1 * fsrc2) - fsrc3);
        break;

    default:
        g_assert_not_reached();
    }

    if (!is_store) {
        result = R(rd);
    } else {
        result = src2;
    }
    if (flag) {
        fprintf(stderr,
                "GPGPU decode lane=%u pc=0x%08x inst=0x%08x opcode = 0x%08x %s "
                "rd=x%u rs1=x%u "
                "rs2=x%u imm=0x%08x result=0x%08x next_pc=0x%08x\n",
                lane->mhartid & MHARTID_THREAD_MASK, lane->pc, inst, opcode,
                opname, rd, rs1, rs2, op_imm, result, next_pc);
    }

    R(0) = 0;
    lane->pc = next_pc;
}

int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    for (uint32_t cycle = 0; cycle < max_cycles; cycle++) {
        bool all_inactive = true;
        for (int lane_id = 0; lane_id < GPGPU_WARP_SIZE; lane_id++) {
            GPGPULane *lane = &warp->lanes[lane_id];
            if (lane->active) {
                all_inactive = false;
                uint32_t inst = vmem_read(s, lane->pc, 4);
                decode(s, lane, inst);
            }
        }
        if (all_inactive) {
            return 0; /* Warp 执行完成 */
        }
    }
    return -1; /* Warp 执行超时 */
}

int gpgpu_core_exec_kernel(GPGPUState *s)
{
    GPGPUKernelParams kernel = gpgpu_core_kernel(s);
    uint64_t num_threads = (uint64_t)kernel.block_dim[0] *
                           kernel.block_dim[1] * kernel.block_dim[2];
    if (num_threads == 0) {
        return 0;
    }
    if (num_threads > GPGPU_WARP_SIZE) {
        num_threads = GPGPU_WARP_SIZE;
    }

    GPGPUWarp *warp = g_malloc0(sizeof(GPGPUWarp));
    gpgpu_core_init_warp(warp, kernel.kernel_addr, 0, kernel.block_dim,
                         num_threads, 0, 0);
    int ret = gpgpu_core_exec_warp(s, warp, 1000000);
    g_free(warp);
    return ret;
}
