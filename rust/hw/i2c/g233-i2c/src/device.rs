use std::ffi::CStr;

use bql::BqlRefCell;
use common::prelude::*;
use hwcore::{prelude::*, DevicePropertiesImpl, DeviceState, IRQState};
use i2c_rs::I2CBus;
use migration::prelude::*;
use qom::{ObjectImpl, ObjectType, Owned, ParentField, ParentInit, prelude::*};
use system::{prelude::*, hwaddr, MemoryRegion, MemoryRegionOps, MemoryRegionOpsBuilder, SysBusDevice};
use util::prelude::*;
use crate::at24c02::At24c02;

const AT24C02_ADDR: u8 = 0x50;

const I2C_CTRL_EN: u32 = 1u32 << 0;
const I2C_CTRL_START: u32 = 1u32 << 1;
const I2C_CTRL_STOP: u32 = 1u32 << 2;
const I2C_CTRL_RW: u32 = 1u32 << 3; // 0=write, 1=read

const I2C_ST_BUSY: u32 = 1u32 << 0;
const I2C_ST_ACK: u32 = 1u32 << 1;
const I2C_ST_DONE: u32 = 1u32 << 2;


#[repr(C)]
#[derive(Debug, Default)]
pub struct G233i2cRegisters {
    pub i2c_ctrl: u32,
    pub i2c_status: u32,
    pub i2c_addr: u32,
    pub i2c_data: u32,
    pub i2c_prescale: u32,
}

#[repr(u64)]
#[derive(Debug, Eq, PartialEq, common::TryInto, Copy, Clone)]
pub enum RegisterOffset {
    CTRL = 0x00,
    STATUS = 0x04,
    ADDR = 0x08,
    DATA = 0x0c,
    PRESCALE = 0x10,
}

#[repr(C)]
#[derive(qom::Object)]
pub struct G233i2cState {
    parent_obj: ParentField<SysBusDevice>,
    pub iomem: MemoryRegion,
    pub regs: BqlRefCell<G233i2cRegisters>,
    pub i2c_bus: BqlRefCell<I2CBus>,
}

qom_isa!(G233i2cState: SysBusDevice, DeviceState, Object);

unsafe impl ObjectType for G233i2cState {
    type Class = <SysBusDevice as ObjectType>::Class;
    const TYPE_NAME: &'static CStr = c"G233i2cState";
}

impl ObjectImpl for G233i2cState {
    type ParentType = SysBusDevice;

    const INSTANCE_INIT: Option<unsafe fn(ParentInit<Self>)> = Some(Self::init);
    const INSTANCE_POST_INIT: Option<fn(&Self)> = Some(Self::post_init);
    const CLASS_INIT: fn(&mut Self::Class) = Self::Class::class_init::<Self>;
}

impl DeviceImpl for G233i2cState {
    const VMSTATE: Option<VMStateDescription<Self>> = None;
    const REALIZE: Option<fn(&Self) -> util::Result<()>> = Some(Self::realize);
}

unsafe impl DevicePropertiesImpl for G233i2cState {}

impl ResettablePhasesImpl for G233i2cState {
    const HOLD: Option<fn(&Self, ResetType)> = Some(Self::reset_hold);
}

impl SysBusDeviceImpl for G233i2cState {}

impl G233i2cState {
    fn set_status_bit(status: &mut u32, mask: u32, on: bool) {
        if on {
            *status |= mask;
        } else {
            *status &= !mask;
        }
    }

    unsafe fn init(mut this: ParentInit<Self>) {
        static G233I2C_OPS: MemoryRegionOps<G233i2cState> =
            MemoryRegionOpsBuilder::<G233i2cState>::new()
                .read(&G233i2cState::read)
                .write(&G233i2cState::write)
                .little_endian()
                .impl_sizes(4, 4)
                .build();

        MemoryRegion::init_io(
            &mut uninit_field_mut!(*this, iomem),
            &G233I2C_OPS,
            "G233I2C_OPS-mmio",
            0x20,
        );

        uninit_field_mut!(*this, regs).write(Default::default());
        uninit_field_mut!(*this, i2c_bus).write(BqlRefCell::new(I2CBus::new()));
    }

    fn post_init(&self) {
        self.init_mmio(&self.iomem);
    }

    fn read(&self, offset: hwaddr, _size: u32) -> u64 {
        match RegisterOffset::try_from(offset) {
            Ok(field) => self.regs.borrow_mut().read(field).into(),
            Err(_) => {
                log_mask_ln!(Log::GuestError, "G233i2cState::read: Bad offset {offset}");
                0
            }
        }
    }

    fn write(&self, offset: hwaddr, value: u64, _size: u32) {
        if let Ok(field) = RegisterOffset::try_from(offset) {
            let value = value as u32;
            self.regs.borrow_mut().write(field, value);

            if field != RegisterOffset::CTRL {
                return;
            }

            let mut bus = self.i2c_bus.borrow_mut();
            let mut regs = self.regs.borrow_mut();

            // If the controller is disabled, clear busy and ack, set done, and return.
            if (value & I2C_CTRL_EN) == 0 {
                regs.i2c_status &= !(I2C_ST_BUSY | I2C_ST_ACK);
                regs.i2c_status |= I2C_ST_DONE;
                return;
            }

            regs.i2c_status &= !I2C_ST_DONE;
            let is_recv = (value & I2C_CTRL_RW) != 0;

            // If START is set, begin a transfer.
            if (value & I2C_CTRL_START) != 0 {
                let addr = (regs.i2c_addr & 0x7f) as u8;
                let ack = bus.start_transfer(addr, is_recv) == 0;
                Self::set_status_bit(&mut regs.i2c_status, I2C_ST_ACK, ack);
                Self::set_status_bit(&mut regs.i2c_status, I2C_ST_BUSY, bus.is_busy());
                regs.i2c_status |= I2C_ST_DONE;
                return;
            }

            // If STOP is set, end the transfer and set status bits accordingly.
            if (value & I2C_CTRL_STOP) != 0 {
                bus.end_transfer();
                regs.i2c_status &= !(I2C_ST_BUSY | I2C_ST_ACK);
                regs.i2c_status |= I2C_ST_DONE;
                return;
            }

            // If not START or STOP, we must be in the middle of a transfer.
            let ack = if is_recv {
                regs.i2c_data = u32::from(bus.recv());
                true
            } else {
                bus.send((regs.i2c_data & 0xff) as u8) == 0
            };

            Self::set_status_bit(&mut regs.i2c_status, I2C_ST_ACK, ack);
            Self::set_status_bit(&mut regs.i2c_status, I2C_ST_BUSY, bus.is_busy());
            regs.i2c_status |= I2C_ST_DONE;
        } else {
            log_mask_ln!(
                Log::GuestError,
                "G233i2cState::write: Bad offset {offset} value {value}"
            );
        }
    }

    fn realize(&self) -> util::Result<()> {
        let mut bus = self.i2c_bus.borrow_mut();
        if bus.device_count() == 0 {
            bus.attach(Box::new(At24c02::new(AT24C02_ADDR)));
        }
        Ok(())
    }

    fn reset_hold(&self, _type: ResetType) {}
}

impl G233i2cRegisters {
    pub(self) fn read(&mut self, offset: RegisterOffset) -> u32 {
        use RegisterOffset::*;
        match offset {
            CTRL => self.i2c_ctrl,
            STATUS => self.i2c_status,
            ADDR => self.i2c_addr,
            DATA => self.i2c_data,
            PRESCALE => self.i2c_prescale,
        }
    }

    pub(self) fn write(&mut self, offset: RegisterOffset, value: u32) {
        use RegisterOffset::*;
        match offset {
            CTRL => self.i2c_ctrl = value,
            STATUS => self.i2c_status = value,
            ADDR => self.i2c_addr = value,
            DATA => self.i2c_data = value,
            PRESCALE => self.i2c_prescale = value,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn G233i2c_create(
    addr: u64,
    irq: *mut IRQState,
) -> *mut DeviceState {
    let dev = G233i2cState::new();

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
