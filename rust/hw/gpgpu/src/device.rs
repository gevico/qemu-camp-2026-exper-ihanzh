#![allow(dead_code)]

use std::cell::Cell;
use std::ffi::CStr;
use std::os::raw::c_int;
use std::pin::Pin;
use std::ptr;

use common::prelude::*;
use hwcore::{prelude::*, DeviceState};
use migration::prelude::*;
use qom::{prelude::*, ObjectImpl, ObjectType, ParentField, ParentInit};
use system::{
    hwaddr, prelude::*, MemoryRegion, MemoryRegionOps, MemoryRegionOpsBuilder, PCIDevice,
};
use util::prelude::*;

const GPGPU_CTRL_BAR_SIZE: u64 = 1 * 1024 * 1024;
const GPGPU_VRAM_BAR_SIZE: u64 = 64 * 1024 * 1024;
const GPGPU_DOORBELL_BAR_SIZE: u64 = 64 * 1024;
const GPGPU_DEFAULT_NUM_CUS: u32 = 4;
const GPGPU_DEFAULT_WARPS_PER_CU: u32 = 4;
const GPGPU_DEFAULT_WARP_SIZE: u32 = 32;
const GPGPU_DEFAULT_VRAM_SIZE: u64 = 64 * 1024 * 1024;
const GPGPU_DEV_ID_VALUE: u32 = 0x47505055;
const GPGPU_DEV_VERSION_VALUE: u32 = 0x00010000;
const GPGPU_VENDOR_ID: u16 = 0x1234;
const GPGPU_DEVICE_ID: u16 = 0x1337;
const GPGPU_REVISION: u8 = 0x01;
const GPGPU_CLASS_CODE: u16 = 0x0302;
const PCI_BASE_ADDRESS_SPACE_MEMORY: u8 = 0x00;
const PCI_BASE_ADDRESS_MEM_TYPE_64: u8 = 0x04;
const PCI_BASE_ADDRESS_MEM_PREFETCH: u8 = 0x08;
const GPGPU_STATUS_READY: u32 = 1;
const GPGPU_STATUS_BUSY: u32 = 1 << 1;
const GPGPU_STATUS_ERROR: u32 = 1 << 2;
const GPGPU_ERR_VRAM_FAULT: u32 = 1 << 1;
const GPGPU_ERR_KERNEL_FAULT: u32 = 1 << 2;

const GPGPU_DMA_CTRL_START: u32 = 1 << 0;
const GPGPU_DMA_CTRL_DIR_TO_VRAM: u32 = 0 << 1;
const GPGPU_DMA_CTRL_DIR_FROM_VRAM: u32 = 1 << 1;
const GPGPU_DMA_CTRL_IRQ_ENABLE: u32 = 1 << 2;

const GPGPU_DMA_STATUS_BUSY: u32 = 1 << 0;
const GPGPU_DMA_STATUS_COMPLETE: u32 = 1 << 1;
const GPGPU_DMA_STATUS_ERROR: u32 = 1 << 2;

const GPGPU_REG_DEV_ID: hwaddr = 0x0000;
const GPGPU_REG_DEV_VERSION: hwaddr = 0x0004;
const GPGPU_REG_VRAM_SIZE_LO: hwaddr = 0x000c;
const GPGPU_REG_VRAM_SIZE_HI: hwaddr = 0x0010;
const GPGPU_REG_GLOBAL_CTRL: hwaddr = 0x0100;
const GPGPU_REG_GLOBAL_STATUS: hwaddr = 0x0104;
const GPGPU_REG_ERROR_STATUS: hwaddr = 0x0108;
const GPGPU_REG_IRQ_ENABLE: hwaddr = 0x0200;
const GPGPU_REG_IRQ_STATUS: hwaddr = 0x0204;
const GPGPU_REG_SHARED_MEM_SIZE: hwaddr = 0x0328;
const GPGPU_REG_GRID_DIM_X: hwaddr = 0x0310;
const GPGPU_REG_GRID_DIM_Y: hwaddr = 0x0314;
const GPGPU_REG_GRID_DIM_Z: hwaddr = 0x0318;
const GPGPU_REG_BLOCK_DIM_X: hwaddr = 0x031c;
const GPGPU_REG_BLOCK_DIM_Y: hwaddr = 0x0320;
const GPGPU_REG_BLOCK_DIM_Z: hwaddr = 0x0324;
const GPGPU_REG_KERNEL_ADDR_LO: hwaddr = 0x0300;
const GPGPU_REG_KERNEL_ADDR_HI: hwaddr = 0x0304;
const GPGPU_REG_KERNEL_ARGS_LO: hwaddr = 0x0308;
const GPGPU_REG_KERNEL_ARGS_HI: hwaddr = 0x030c;
const GPGPU_REG_DISPATCH: hwaddr = 0x0330;
const GPGPU_REG_DMA_SRC_LO: hwaddr = 0x0400;
const GPGPU_REG_DMA_SRC_HI: hwaddr = 0x0404;
const GPGPU_REG_DMA_DST_LO: hwaddr = 0x0408;
const GPGPU_REG_DMA_DST_HI: hwaddr = 0x040c;
const GPGPU_REG_DMA_SIZE: hwaddr = 0x0410;
const GPGPU_REG_DMA_CTRL: hwaddr = 0x0414;
const GPGPU_REG_DMA_STATUS: hwaddr = 0x0418;
const GPGPU_REG_THREAD_ID_X: hwaddr = 0x1000;
const GPGPU_REG_THREAD_ID_Y: hwaddr = 0x1004;
const GPGPU_REG_THREAD_ID_Z: hwaddr = 0x1008;
const GPGPU_REG_BLOCK_ID_X: hwaddr = 0x1010;
const GPGPU_REG_BLOCK_ID_Y: hwaddr = 0x1014;
const GPGPU_REG_BLOCK_ID_Z: hwaddr = 0x1018;
const GPGPU_REG_WARP_ID: hwaddr = 0x1020;
const GPGPU_REG_LANE_ID: hwaddr = 0x1024;
const GPGPU_REG_THREAD_MASK: hwaddr = 0x2004;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GPGPUKernelParams {
    pub kernel_addr: u64,     /* 内核代码在 VRAM 中的地址 */
    pub kernel_args: u64,     /* 内核参数在 VRAM 中的地址 */
    pub grid_dim: [u32; 3],   /* Grid 维度 [X, Y, Z] */
    pub block_dim: [u32; 3],  /* Block 维度 [X, Y, Z] */
    pub shared_mem_size: u32, /* 每个 Block 的共享内存大小 */
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GPGPUDMAState {
    pub src_addr: u64, /* 源地址 */
    pub dst_addr: u64, /* 目标地址 */
    pub size: u32,     /* 传输大小 */
    pub ctrl: u32,     /* 控制寄存器值 */
    pub status: u32,   /* 状态寄存器值 */
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct GPGPUSIMTContext {
    /* 当前执行的线程位置 */
    pub thread_id: [u32; 3], /* threadIdx.x/y/z */
    pub block_id: [u32; 3],  /* blockIdx.x/y/z */
    pub warp_id: u32,        /* 当前 warp ID */
    pub lane_id: u32,        /* 线程在 warp 中的位置 (0-31) */

    /* Barrier 同步状态 */
    pub barrier_count: u32,   /* 到达 barrier 的线程数 */
    pub barrier_target: u32,  /* 需要到达的线程总数 */
    pub barrier_active: bool, /* barrier 是否激活 */

    /* 活跃线程掩码 (用于分支分歧) */
    pub thread_mask: u32, /* 32 位掩码，每位代表一个线程 */
}

#[repr(C)]
#[derive(qom::Object, hwcore::Device)]
pub struct GPGPUState {
    parent_obj: ParentField<PCIDevice>,
    pub ctrl_mmio: MemoryRegion,
    pub vram: MemoryRegion,
    pub doorbell_mmio: MemoryRegion,

    #[property(default = GPGPU_DEFAULT_NUM_CUS)]
    pub num_cus: u32,
    #[property(default = GPGPU_DEFAULT_WARPS_PER_CU)]
    pub warps_per_cu: u32,
    #[property(default = GPGPU_DEFAULT_WARP_SIZE)]
    pub warp_size: u32,
    #[property(default = GPGPU_DEFAULT_VRAM_SIZE)]
    pub vram_size: u64,
    pub vram_ptr: Cell<*mut u8>,

    pub global_ctrl: Cell<u32>,
    pub global_status: Cell<u32>,
    pub error_status: Cell<u32>,

    pub irq_enable: Cell<u32>,
    pub irq_status: Cell<u32>,

    pub kernel: Cell<GPGPUKernelParams>,

    pub dma: Cell<GPGPUDMAState>,
    pub dma_timer: Timer,

    pub kernel_timer: Timer,

    pub simt: Cell<GPGPUSIMTContext>,
}

qom_isa!(GPGPUState: PCIDevice, DeviceState, Object);

unsafe impl Sync for GPGPUState {}

unsafe impl ObjectType for GPGPUState {
    type Class = <PCIDevice as ObjectType>::Class;
    const TYPE_NAME: &'static CStr = c"gpgpu";
}

impl ObjectImpl for GPGPUState {
    type ParentType = PCIDevice;

    const INSTANCE_INIT: Option<unsafe fn(ParentInit<Self>)> = Some(Self::init);
    const TYPE_INFO: qom::bindings::TypeInfo = qom::bindings::TypeInfo {
        name: Self::TYPE_NAME.as_ptr(),
        parent: <Self::ParentType as ObjectType>::TYPE_NAME.as_ptr(),
        instance_size: core::mem::size_of::<Self>(),
        instance_align: core::mem::align_of::<Self>(),
        instance_init: Some(qom::rust_instance_init::<Self>),
        instance_post_init: None,
        instance_finalize: Some(qom::drop_object::<Self>),
        abstract_: false,
        class_size: core::mem::size_of::<Self::Class>(),
        class_init: Some(qom::rust_class_init::<Self>),
        class_base_init: None,
        class_data: core::ptr::null(),
        interfaces: (&[
            qom::bindings::InterfaceInfo {
                type_: system::bindings::INTERFACE_CONVENTIONAL_PCI_DEVICE
                    .as_ptr()
                    .cast(),
            },
            qom::bindings::InterfaceInfo {
                type_: core::ptr::null(),
            },
        ] as *const [qom::bindings::InterfaceInfo; 2])
            .cast(),
    };
    const CLASS_INIT: fn(&mut Self::Class) = Self::Class::class_init::<Self>;
}

impl DeviceImpl for GPGPUState {
    const VMSTATE: Option<VMStateDescription<Self>> = None;
}

impl PCIDeviceImpl for GPGPUState {
    const REALIZE: Option<fn(&Self) -> util::Result<()>> = Some(Self::realize);
    const EXIT: Option<fn(&Self)> = Some(Self::exit);
    const VENDOR_ID: u16 = GPGPU_VENDOR_ID;
    const DEVICE_ID: u16 = GPGPU_DEVICE_ID;
    const REVISION: u8 = GPGPU_REVISION;
    const CLASS_ID: u16 = GPGPU_CLASS_CODE;
}

impl ResettablePhasesImpl for GPGPUState {
    const HOLD: Option<fn(&Self, ResetType)> = Some(Self::reset_hold);
}

unsafe extern "C" {
    fn gpgpu_core_exec_kernel(s: *mut GPGPUState) -> c_int;
}

impl GPGPUState {
    fn vram_size(&self) -> u64 {
        self.vram_size
    }

    fn reset_state(&self) {
        self.global_ctrl.set(0);
        self.global_status.set(GPGPU_STATUS_READY);
        self.error_status.set(0);
        self.irq_enable.set(0);
        self.irq_status.set(0);
        self.kernel.set(GPGPUKernelParams::default());
        self.dma.set(GPGPUDMAState::default());
        self.simt.set(GPGPUSIMTContext::default());
        self.dma_timer.delete();
        self.kernel_timer.delete();

        let vram_ptr = self.vram_ptr.get();
        if !vram_ptr.is_null() {
            let size = self.vram_size() as usize;
            unsafe {
                ptr::write_bytes(vram_ptr, 0, size);
            }
        }
    }

    fn vram_access_ok(&self, offset: hwaddr, size: u32, op: &'static str) -> bool {
        let size_u64 = u64::from(size);
        let end = match offset.checked_add(size_u64) {
            Some(end) => end,
            None => {
                self.error_status
                    .set(self.error_status.get() | GPGPU_ERR_VRAM_FAULT);
                self.global_status
                    .set(self.global_status.get() | GPGPU_STATUS_ERROR);
                log_mask_ln!(
                    Log::GuestError,
                    "GPGPUState::{op}: access overflow at {offset:#x}"
                );
                return false;
            }
        };

        if end > self.vram_size() {
            self.error_status
                .set(self.error_status.get() | GPGPU_ERR_VRAM_FAULT);
            self.global_status
                .set(self.global_status.get() | GPGPU_STATUS_ERROR);
            log_mask_ln!(
                Log::GuestError,
                "GPGPUState::{op}: out-of-range access offset={offset:#x} size={size}"
            );
            return false;
        }

        true
    }

    unsafe fn init(mut this: ParentInit<Self>) {
        static GPGPU_CTRL_OPS: MemoryRegionOps<GPGPUState> =
            MemoryRegionOpsBuilder::<GPGPUState>::new()
                .read(&GPGPUState::read)
                .write(&GPGPUState::write)
                .little_endian()
                .impl_sizes(4, 4)
                .build();

        static GPGPU_VRAM_OPS: MemoryRegionOps<GPGPUState> =
            MemoryRegionOpsBuilder::<GPGPUState>::new()
                .read(&GPGPUState::vram_read)
                .write(&GPGPUState::vram_write)
                .little_endian()
                .impl_sizes(1, 8)
                .build();

        static GPGPU_DOORBELL_OPS: MemoryRegionOps<GPGPUState> =
            MemoryRegionOpsBuilder::<GPGPUState>::new()
                .read(&GPGPUState::doorbell_read)
                .write(&GPGPUState::doorbell_write)
                .little_endian()
                .impl_sizes(4, 4)
                .build();

        MemoryRegion::init_io(
            &mut uninit_field_mut!(*this, ctrl_mmio),
            &GPGPU_CTRL_OPS,
            "gpgpu-ctrl",
            GPGPU_CTRL_BAR_SIZE,
        );

        MemoryRegion::init_io(
            &mut uninit_field_mut!(*this, vram),
            &GPGPU_VRAM_OPS,
            "gpgpu-vram",
            GPGPU_VRAM_BAR_SIZE,
        );

        MemoryRegion::init_io(
            &mut uninit_field_mut!(*this, doorbell_mmio),
            &GPGPU_DOORBELL_OPS,
            "gpgpu-doorbell",
            GPGPU_DOORBELL_BAR_SIZE,
        );

        uninit_field_mut!(*this, num_cus).write(GPGPU_DEFAULT_NUM_CUS);
        uninit_field_mut!(*this, warps_per_cu).write(GPGPU_DEFAULT_WARPS_PER_CU);
        uninit_field_mut!(*this, warp_size).write(GPGPU_DEFAULT_WARP_SIZE);
        uninit_field_mut!(*this, vram_size).write(GPGPU_DEFAULT_VRAM_SIZE);
        uninit_field_mut!(*this, vram_ptr).write(Cell::new(ptr::null_mut()));
        uninit_field_mut!(*this, global_ctrl).write(Cell::new(0));
        uninit_field_mut!(*this, global_status).write(Cell::new(0));
        uninit_field_mut!(*this, error_status).write(Cell::new(0));
        uninit_field_mut!(*this, irq_enable).write(Cell::new(0));
        uninit_field_mut!(*this, irq_status).write(Cell::new(0));
        uninit_field_mut!(*this, kernel).write(Cell::new(GPGPUKernelParams::default()));
        uninit_field_mut!(*this, dma).write(Cell::new(GPGPUDMAState::default()));
        uninit_field_mut!(*this, simt).write(Cell::new(GPGPUSIMTContext::default()));
        uninit_field_mut!(*this, dma_timer).write(unsafe { Timer::new() });
        uninit_field_mut!(*this, kernel_timer).write(unsafe { Timer::new() });

        Timer::init_full(
            unsafe { Pin::new_unchecked(&mut *this.as_mut_ptr()) },
            None,
            CLOCK_VIRTUAL,
            Timer::NS,
            0,
            GPGPUState::dma_timer_cb,
            |s| &mut s.dma_timer,
        );

        Timer::init_full(
            unsafe { Pin::new_unchecked(&mut *this.as_mut_ptr()) },
            None,
            CLOCK_VIRTUAL,
            Timer::NS,
            0,
            GPGPUState::kernel_timer_cb,
            |s| &mut s.kernel_timer,
        );
    }

    fn kernel_timer_cb(_state: &GPGPUState) {}

    fn dma_timer_cb(_state: &GPGPUState) {}

    unsafe fn exec_kernel(&self) -> i32 {
        // SAFETY: the C core only sees an opaque pointer and calls back
        // into Rust through exported accessors.
        unsafe { gpgpu_core_exec_kernel(self as *const _ as *mut GPGPUState) }
    }

    fn read(&self, offset: hwaddr, _size: u32) -> u64 {
        let kernel = self.kernel.get();
        let dma = self.dma.get();
        let simt = self.simt.get();

        match offset {
            GPGPU_REG_DEV_ID => GPGPU_DEV_ID_VALUE as u64,
            GPGPU_REG_DEV_VERSION => GPGPU_DEV_VERSION_VALUE as u64,
            GPGPU_REG_VRAM_SIZE_LO => (self.vram_size() & 0xffff_ffff) as u64,
            GPGPU_REG_VRAM_SIZE_HI => (self.vram_size() >> 32) as u64,
            GPGPU_REG_GLOBAL_CTRL => self.global_ctrl.get() as u64,
            GPGPU_REG_GLOBAL_STATUS => self.global_status.get() as u64,
            GPGPU_REG_ERROR_STATUS => self.error_status.get() as u64,
            GPGPU_REG_IRQ_ENABLE => self.irq_enable.get() as u64,
            GPGPU_REG_IRQ_STATUS => self.irq_status.get() as u64,
            GPGPU_REG_SHARED_MEM_SIZE => kernel.shared_mem_size as u64,
            GPGPU_REG_GRID_DIM_X => kernel.grid_dim[0] as u64,
            GPGPU_REG_GRID_DIM_Y => kernel.grid_dim[1] as u64,
            GPGPU_REG_GRID_DIM_Z => kernel.grid_dim[2] as u64,
            GPGPU_REG_BLOCK_DIM_X => kernel.block_dim[0] as u64,
            GPGPU_REG_BLOCK_DIM_Y => kernel.block_dim[1] as u64,
            GPGPU_REG_BLOCK_DIM_Z => kernel.block_dim[2] as u64,
            GPGPU_REG_KERNEL_ADDR_LO => (kernel.kernel_addr & 0xffff_ffff) as u64,
            GPGPU_REG_KERNEL_ADDR_HI => (kernel.kernel_addr >> 32) as u64,
            GPGPU_REG_KERNEL_ARGS_LO => (kernel.kernel_args & 0xffff_ffff) as u64,
            GPGPU_REG_KERNEL_ARGS_HI => (kernel.kernel_args >> 32) as u64,
            GPGPU_REG_DMA_SRC_LO => (dma.src_addr & 0xffff_ffff) as u64,
            GPGPU_REG_DMA_SRC_HI => (dma.src_addr >> 32) as u64,
            GPGPU_REG_DMA_DST_LO => (dma.dst_addr & 0xffff_ffff) as u64,
            GPGPU_REG_DMA_DST_HI => (dma.dst_addr >> 32) as u64,
            GPGPU_REG_DMA_SIZE => dma.size as u64,
            GPGPU_REG_DMA_CTRL => (dma.ctrl & 0b111) as u64,
            GPGPU_REG_DMA_STATUS => (dma.status & 0b111) as u64,
            GPGPU_REG_THREAD_ID_X => simt.thread_id[0] as u64,
            GPGPU_REG_THREAD_ID_Y => simt.thread_id[1] as u64,
            GPGPU_REG_THREAD_ID_Z => simt.thread_id[2] as u64,
            GPGPU_REG_BLOCK_ID_X => simt.block_id[0] as u64,
            GPGPU_REG_BLOCK_ID_Y => simt.block_id[1] as u64,
            GPGPU_REG_BLOCK_ID_Z => simt.block_id[2] as u64,
            GPGPU_REG_WARP_ID => simt.warp_id as u64,
            GPGPU_REG_LANE_ID => simt.lane_id as u64,
            GPGPU_REG_THREAD_MASK => simt.thread_mask as u64,
            _ => {
                log_mask_ln!(Log::GuestError, "GPGPUState::read: Bad offset {offset}");
                0
            }
        }
    }

    fn write(&self, offset: hwaddr, value: u64, _size: u32) {
        match offset {
            GPGPU_REG_GLOBAL_CTRL => {
                let ctrl = value as u32;
                self.global_ctrl.set(ctrl);
                if (ctrl >> 1) & 0x1 != 0 {
                    self.reset_state();
                }
            }
            GPGPU_REG_ERROR_STATUS => {
                self.error_status
                    .set(self.error_status.get() & !(value as u32));
            }
            GPGPU_REG_IRQ_ENABLE => {
                self.irq_enable.set((value & 0xffff) as u32);
            }
            GPGPU_REG_IRQ_STATUS => {
                self.irq_status.set(self.irq_status.get() & !(value as u32));
            }
            GPGPU_REG_SHARED_MEM_SIZE => {
                let mut kernel = self.kernel.get();
                kernel.shared_mem_size = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_GRID_DIM_X => {
                let mut kernel = self.kernel.get();
                kernel.grid_dim[0] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_GRID_DIM_Y => {
                let mut kernel = self.kernel.get();
                kernel.grid_dim[1] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_GRID_DIM_Z => {
                let mut kernel = self.kernel.get();
                kernel.grid_dim[2] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_BLOCK_DIM_X => {
                let mut kernel = self.kernel.get();
                kernel.block_dim[0] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_BLOCK_DIM_Y => {
                let mut kernel = self.kernel.get();
                kernel.block_dim[1] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_BLOCK_DIM_Z => {
                let mut kernel = self.kernel.get();
                kernel.block_dim[2] = value as u32;
                self.kernel.set(kernel);
            }
            GPGPU_REG_KERNEL_ADDR_LO => {
                let mut kernel = self.kernel.get();
                kernel.kernel_addr = (kernel.kernel_addr & !0xffff_ffff) | (value & 0xffff_ffff);
                self.kernel.set(kernel);
            }
            GPGPU_REG_KERNEL_ADDR_HI => {
                let mut kernel = self.kernel.get();
                kernel.kernel_addr =
                    (kernel.kernel_addr & 0xffff_ffff) | ((value & 0xffff_ffff) << 32);
                self.kernel.set(kernel);
            }
            GPGPU_REG_KERNEL_ARGS_LO => {
                let mut kernel = self.kernel.get();
                kernel.kernel_args = (kernel.kernel_args & !0xffff_ffff) | (value & 0xffff_ffff);
                self.kernel.set(kernel);
            }
            GPGPU_REG_KERNEL_ARGS_HI => {
                let mut kernel = self.kernel.get();
                kernel.kernel_args =
                    (kernel.kernel_args & 0xffff_ffff) | ((value & 0xffff_ffff) << 32);
                self.kernel.set(kernel);
            }
            GPGPU_REG_DISPATCH => {
                let status = (self.global_status.get() | GPGPU_STATUS_BUSY) & !GPGPU_STATUS_READY;
                self.global_status.set(status);
                let ret = unsafe { self.exec_kernel() };
                if ret < 0 {
                    self.error_status
                        .set(self.error_status.get() | GPGPU_ERR_KERNEL_FAULT);
                    self.global_status
                        .set(self.global_status.get() | GPGPU_STATUS_ERROR);
                }
                self.global_status
                    .set((self.global_status.get() & !GPGPU_STATUS_BUSY) | GPGPU_STATUS_READY);
            }
            GPGPU_REG_DMA_SRC_LO => {
                let mut dma = self.dma.get();
                dma.src_addr = (dma.src_addr & !0xffff_ffff) | (value & 0xffff_ffff);
                self.dma.set(dma);
            }
            GPGPU_REG_DMA_SRC_HI => {
                let mut dma = self.dma.get();
                dma.src_addr = (dma.src_addr & 0xffff_ffff) | ((value & 0xffff_ffff) << 32);
                self.dma.set(dma);
            }
            GPGPU_REG_DMA_DST_LO => {
                let mut dma = self.dma.get();
                dma.dst_addr = (dma.dst_addr & !0xffff_ffff) | (value & 0xffff_ffff);
                self.dma.set(dma);
            }
            GPGPU_REG_DMA_DST_HI => {
                let mut dma = self.dma.get();
                dma.dst_addr = (dma.dst_addr & 0xffff_ffff) | ((value & 0xffff_ffff) << 32);
                self.dma.set(dma);
            }
            GPGPU_REG_DMA_SIZE => {
                let mut dma = self.dma.get();
                dma.size = value as u32;
                self.dma.set(dma);
            }
            GPGPU_REG_DMA_CTRL => {
                let mut dma = self.dma.get();
                dma.ctrl = (value & 0b111) as u32;
                self.dma.set(dma);
            }
            GPGPU_REG_THREAD_ID_X => {
                let mut simt = self.simt.get();
                simt.thread_id[0] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_THREAD_ID_Y => {
                let mut simt = self.simt.get();
                simt.thread_id[1] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_THREAD_ID_Z => {
                let mut simt = self.simt.get();
                simt.thread_id[2] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_BLOCK_ID_X => {
                let mut simt = self.simt.get();
                simt.block_id[0] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_BLOCK_ID_Y => {
                let mut simt = self.simt.get();
                simt.block_id[1] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_BLOCK_ID_Z => {
                let mut simt = self.simt.get();
                simt.block_id[2] = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_WARP_ID => {
                let mut simt = self.simt.get();
                simt.warp_id = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_LANE_ID => {
                let mut simt = self.simt.get();
                simt.lane_id = value as u32;
                self.simt.set(simt);
            }
            GPGPU_REG_THREAD_MASK => {
                let mut simt = self.simt.get();
                simt.thread_mask = value as u32;
                self.simt.set(simt);
            }
            _ => {
                log_mask_ln!(
                    Log::GuestError,
                    "GPGPUState::write: Bad offset {offset} value {value}"
                );
            }
        }
    }

    fn vram_read(&self, offset: hwaddr, size: u32) -> u64 {
        let vram_ptr = self.vram_ptr.get();
        if vram_ptr.is_null() {
            log_mask_ln!(Log::GuestError, "GPGPUState::vram_read: vram_ptr is null");
            return 0;
        }
        if !self.vram_access_ok(offset, size, "vram_read") {
            return 0;
        }

        let ptr = unsafe { vram_ptr.add(offset as usize) };
        unsafe {
            match size {
                1 => u64::from(*(ptr as *const u8)),
                2 => u64::from(*(ptr as *const u16)),
                4 => u64::from(*(ptr as *const u32)),
                8 => *(ptr as *const u64),
                _ => {
                    log_mask_ln!(
                        Log::GuestError,
                        "GPGPUState::vram_read: invalid size {size}"
                    );
                    0
                }
            }
        }
    }

    fn vram_write(&self, offset: hwaddr, value: u64, size: u32) {
        let vram_ptr = self.vram_ptr.get();
        if vram_ptr.is_null() {
            log_mask_ln!(Log::GuestError, "GPGPUState::vram_write: vram_ptr is null");
            return;
        }
        if !self.vram_access_ok(offset, size, "vram_write") {
            return;
        }

        let ptr = unsafe { vram_ptr.add(offset as usize) };
        unsafe {
            match size {
                1 => *(ptr as *mut u8) = value as u8,
                2 => *(ptr as *mut u16) = value as u16,
                4 => *(ptr as *mut u32) = value as u32,
                8 => *(ptr as *mut u64) = value as u64,
                _ => log_mask_ln!(
                    Log::GuestError,
                    "GPGPUState::vram_write: invalid size {size}"
                ),
            }
        }
    }

    fn doorbell_read(&self, _offset: hwaddr, _size: u32) -> u64 {
        0
    }

    fn doorbell_write(&self, _offset: hwaddr, _value: u64, _size: u32) {}

    fn realize(&self) -> util::Result<()> {
        ensure!(
            self.vram_size() <= GPGPU_VRAM_BAR_SIZE,
            "gpgpu.vram_size must not exceed the {}-byte VRAM BAR",
            GPGPU_VRAM_BAR_SIZE
        );
        ensure!(
            self.vram_size() <= usize::MAX as u64,
            "gpgpu.vram_size is too large"
        );
        let size = self.vram_size() as usize;
        let vram = vec![0u8; size].into_boxed_slice();
        let vram_ptr = vram.as_ptr().cast_mut();
        let _ = Box::leak(vram);

        unsafe {
            let pci = self.upcast::<PCIDevice>().as_mut_ptr();
            let pci_conf = (*pci).config;
            *pci_conf.add(0x3d) = 1;
        }

        self.vram_ptr.set(vram_ptr);
        self.register_bar(
            0,
            PCI_BASE_ADDRESS_SPACE_MEMORY | PCI_BASE_ADDRESS_MEM_TYPE_64,
            &self.ctrl_mmio,
        );
        self.register_bar(
            2,
            PCI_BASE_ADDRESS_SPACE_MEMORY
                | PCI_BASE_ADDRESS_MEM_TYPE_64
                | PCI_BASE_ADDRESS_MEM_PREFETCH,
            &self.vram,
        );
        self.register_bar(4, PCI_BASE_ADDRESS_SPACE_MEMORY, &self.doorbell_mmio);
        self.reset_state();

        Ok(())
    }

    fn exit(&self) {
        self.dma_timer.delete();
        self.kernel_timer.delete();
        self.vram_ptr.set(ptr::null_mut());
    }

    fn reset_hold(&self, _type: ResetType) {
        self.reset_state();
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpgpu_get_vram_size(state: *const GPGPUState) -> u64 {
    unsafe { (*state).vram_size() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpgpu_get_vram_ptr(state: *const GPGPUState) -> *mut u8 {
    unsafe { (*state).vram_ptr.get() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn gpgpu_get_kernel(state: *const GPGPUState) -> GPGPUKernelParams {
    unsafe { (*state).kernel.get() }
}
