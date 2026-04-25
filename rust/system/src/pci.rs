// SPDX-License-Identifier: GPL-2.0-or-later

//! Bindings to access PCI device functionality from Rust.

use std::{ffi::CStr, ptr::NonNull};

use common::Opaque;
use hwcore::{prelude::*, DeviceState};
use qom::prelude::*;
pub use system_sys::PCIDeviceClass;
use util::{Error, Result};

use crate::MemoryRegion;

/// A safe wrapper around [`system_sys::PCIDevice`].
#[repr(transparent)]
#[derive(Debug, common::Wrapper)]
pub struct PCIDevice(Opaque<system_sys::PCIDevice>);

unsafe impl Send for PCIDevice {}
unsafe impl Sync for PCIDevice {}

unsafe impl ObjectType for PCIDevice {
    type Class = PCIDeviceClass;
    const TYPE_NAME: &'static CStr =
        unsafe { CStr::from_bytes_with_nul_unchecked(system_sys::TYPE_PCI_DEVICE) };
}

qom_isa!(PCIDevice: DeviceState, Object);

pub trait PCIDeviceImpl: DeviceImpl + IsA<PCIDevice> {
    const REALIZE: Option<fn(&Self) -> Result<()>> = None;
    const EXIT: Option<fn(&Self)> = None;
    const VENDOR_ID: u16;
    const DEVICE_ID: u16;
    const REVISION: u8 = 0;
    const CLASS_ID: u16 = 0;
    const SUBSYSTEM_VENDOR_ID: u16 = 0;
    const SUBSYSTEM_ID: u16 = 0;
}

unsafe extern "C" fn rust_pci_realize_fn<T: PCIDeviceImpl>(
    dev: *mut system_sys::PCIDevice,
    errp: *mut *mut util::bindings::Error,
) {
    let state = NonNull::new(dev).unwrap().cast::<T>();
    let result = <T as PCIDeviceImpl>::REALIZE.unwrap()(unsafe { state.as_ref() });
    unsafe {
        Error::ok_or_propagate(result, errp);
    }
}

unsafe extern "C" fn rust_pci_exit_fn<T: PCIDeviceImpl>(dev: *mut system_sys::PCIDevice) {
    let state = NonNull::new(dev).unwrap().cast::<T>();
    <T as PCIDeviceImpl>::EXIT.unwrap()(unsafe { state.as_ref() });
}

pub trait PCIDeviceClassExt {
    fn class_init<T: PCIDeviceImpl>(&mut self);
}

impl PCIDeviceClassExt for PCIDeviceClass {
    fn class_init<T: PCIDeviceImpl>(&mut self) {
        if <T as PCIDeviceImpl>::REALIZE.is_some() {
            self.realize = Some(rust_pci_realize_fn::<T>);
        }
        if <T as PCIDeviceImpl>::EXIT.is_some() {
            self.exit = Some(rust_pci_exit_fn::<T>);
        }

        self.vendor_id = T::VENDOR_ID;
        self.device_id = T::DEVICE_ID;
        self.revision = T::REVISION;
        self.class_id = T::CLASS_ID;
        self.subsystem_vendor_id = T::SUBSYSTEM_VENDOR_ID;
        self.subsystem_id = T::SUBSYSTEM_ID;
        self.parent_class.class_init::<T>();
    }
}

pub trait PCIDeviceMethods: ObjectDeref
where
    Self::Target: IsA<PCIDevice>,
{
    fn register_bar(&self, region_num: u32, ty: u8, memory: &MemoryRegion) {
        assert!(bql::is_locked());
        let region_num: i32 = region_num.try_into().unwrap();
        unsafe {
            system_sys::pci_register_bar(
                self.upcast().as_mut_ptr(),
                region_num,
                ty,
                memory.as_mut_ptr(),
            );
        }
    }
}

impl<R: ObjectDeref> PCIDeviceMethods for R where R::Target: IsA<PCIDevice> {}
