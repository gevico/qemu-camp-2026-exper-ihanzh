use std::ffi::CStr;

use bql::BqlRefCell;
use common::prelude::*;
use hwcore::{prelude::*, DevicePropertiesImpl, DeviceState, IRQState};
use migration::prelude::*;
use qom::{ObjectImpl, ObjectType, Owned, ParentField, ParentInit, prelude::*};
use system::{prelude::*, hwaddr, MemoryRegion, MemoryRegionOps, MemoryRegionOpsBuilder, SysBusDevice};
use util::prelude::*;
use crate::at25flash::At25flash;

const RSPI_CR1_SPE: u32 = 1u32 << 0;    // SPI Enable
const RSPI_CR1_MSTR: u32 = 1u32 << 2;   // Master Mode

const RSPI_SR_RXNE: u32 = 1u32 << 0;    // Receive buffer not empty
const RSPI_SR_TXE: u32 = 1u32 << 1;     // Transmit buffer empty
// const RSPI_SR_OVERRUN: u32 = 1u32 << 4; // Overrun error

#[repr(C)]
#[derive(Debug, Default)]
pub struct G233spiRegisters {
    pub rspi_cr1: u32, // control register 1
    pub rspi_sr: u32,  // status register
    pub rspi_dr: u32,  // data register
    pub rspi_cs: u32,  // chip select register
}

#[repr(u64)]
#[derive(Debug, Eq, PartialEq, common::TryInto, Copy, Clone)]
pub enum RegisterOffset {
    CR1 = 0x00,
    SR = 0x04,
    DR = 0x08,
    CS = 0x0c,
}

#[repr(C)]
#[derive(qom::Object)]
pub struct G233spiState {
    parent_obj: ParentField<SysBusDevice>,
    pub iomem: MemoryRegion,
    pub regs: BqlRefCell<G233spiRegisters>,
    flash: BqlRefCell<At25flash>,
}

qom_isa!(G233spiState: SysBusDevice, DeviceState, Object);

unsafe impl ObjectType for G233spiState {
    type Class = <SysBusDevice as ObjectType>::Class;
    const TYPE_NAME: &'static CStr = c"G233spiState";
}

impl ObjectImpl for G233spiState {
    type ParentType = SysBusDevice;

    const INSTANCE_INIT: Option<unsafe fn(ParentInit<Self>)> = Some(Self::init);
    const INSTANCE_POST_INIT: Option<fn(&Self)> = Some(Self::post_init);
    const CLASS_INIT: fn(&mut Self::Class) = Self::Class::class_init::<Self>;
}

impl DeviceImpl for G233spiState {
    const VMSTATE: Option<VMStateDescription<Self>> = None;
    const REALIZE: Option<fn(&Self) -> util::Result<()>> = Some(Self::realize);
}

unsafe impl DevicePropertiesImpl for G233spiState {}

impl ResettablePhasesImpl for G233spiState {
    const HOLD: Option<fn(&Self, ResetType)> = Some(Self::reset_hold);
}

impl SysBusDeviceImpl for G233spiState {}

impl G233spiState {
    unsafe fn init(mut this: ParentInit<Self>) {
        static G233SPI_OPS: MemoryRegionOps<G233spiState> =
            MemoryRegionOpsBuilder::<G233spiState>::new()
                .read(&G233spiState::read)
                .write(&G233spiState::write)
                .little_endian()
                .impl_sizes(4, 4)
                .build();

        MemoryRegion::init_io(
            &mut uninit_field_mut!(*this, iomem),
            &G233SPI_OPS,
            "G233SPI_OPS-mmio",
            0x10,
        );

        uninit_field_mut!(*this, regs).write(Default::default());
        uninit_field_mut!(*this, flash).write(BqlRefCell::new(At25flash::default()));
    }

    fn post_init(&self) {
        self.init_mmio(&self.iomem);
    }

    fn read(&self, offset: hwaddr, _size: u32) -> u64 {
        match RegisterOffset::try_from(offset) {
            Ok(field) => {
                let mut regs = self.regs.borrow_mut();
                let val = regs.read(field);
                if field == RegisterOffset::DR {
                    regs.rspi_sr &= !RSPI_SR_RXNE;
                }
                val.into()
            }
            Err(_) => {
                log_mask_ln!(Log::GuestError, "G233spiState::read: Bad offset {offset}");
                0
            }
        }
    }

    fn write(&self, offset: hwaddr, value: u64, _size: u32) {
        if let Ok(field) = RegisterOffset::try_from(offset) {
            let value = value as u32;
            let mut regs = self.regs.borrow_mut();
            match field {
                RegisterOffset::CR1 => {
                    regs.rspi_cr1 = value;
                    let enabled = (value & RSPI_CR1_SPE) != 0;
                    if enabled {
                        regs.rspi_sr |= RSPI_SR_TXE;
                    } else {
                        regs.rspi_sr &= !(RSPI_SR_TXE | RSPI_SR_RXNE);
                    }
                }
                RegisterOffset::SR => {
                    regs.rspi_sr = value;
                }
                RegisterOffset::DR => {
                    regs.rspi_dr = value;
                    let tx = (value & 0xff) as u8;
                    let enabled = (regs.rspi_cr1 & RSPI_CR1_SPE) != 0;
                    let master = (regs.rspi_cr1 & RSPI_CR1_MSTR) != 0;
                    let cs0 = regs.rspi_cs == 0;

                    let rx = if enabled && master && cs0 {
                        self.flash.borrow_mut().xfer(tx)
                    } else {
                        0
                    };

                    regs.rspi_dr = u32::from(rx);
                    regs.rspi_sr |= RSPI_SR_RXNE;
                    if enabled {
                        regs.rspi_sr |= RSPI_SR_TXE;
                    }
                }
                RegisterOffset::CS => {
                    regs.rspi_cs = value;
                }
            }
        } else {
            log_mask_ln!(
                Log::GuestError,
                "G233spiState::write: Bad offset {offset} value {value}"
            );
        }
    }

    fn realize(&self) -> util::Result<()> {
        Ok(())
    }

    fn reset_hold(&self, _type: ResetType) {}
}

impl G233spiRegisters {
    pub(self) fn read(&mut self, offset: RegisterOffset) -> u32 {
        use RegisterOffset::*;
        match offset {
            CR1 => self.rspi_cr1,
            SR => self.rspi_sr,
            DR => self.rspi_dr,
            CS => self.rspi_cs,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn G233spi_create(
    addr: u64,
    irq: *mut IRQState,
) -> *mut DeviceState {
    let dev = G233spiState::new();

    dev.sysbus_realize().unwrap_fatal();
    dev.mmio_map(0, addr);

    if !irq.is_null() {
        // SAFETY: The caller promises that non-null IRQ pointers are valid.
        let irq = unsafe { Owned::<IRQState>::from(&*irq) };
        dev.connect_irq(0, &irq);
        // The pointer is kept alive by the QOM tree; drop the owned ref
    }

    dev.as_mut_ptr()
}
