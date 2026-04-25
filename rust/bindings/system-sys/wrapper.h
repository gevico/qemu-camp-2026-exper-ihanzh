/* SPDX-License-Identifier: GPL-2.0-or-later */

/*
 * This header file is meant to be used as input to the `bindgen` application
 * in order to generate C FFI compatible Rust bindings.
 */

/*
 * We block include/qemu/typedefs.h from bindgen, add here symbols
 * that are needed as opaque types by other functions.
 */
typedef struct DirtyBitmapSnapshot DirtyBitmapSnapshot;
typedef struct FWCfgState FWCfgState;
typedef struct MemoryRegion MemoryRegion;
typedef struct PCIBus PCIBus;
typedef struct RAMBlock RAMBlock;
typedef struct QEMUFile QEMUFile;
typedef struct VMStateDescription VMStateDescription;
typedef struct SHPCDevice SHPCDevice;

#include "qemu/osdep.h"

#include "exec/hwaddr.h"
#include "system/address-spaces.h"
#include "system/memory.h"
#include "hw/core/sysbus.h"
#include "hw/pci/pci.h"
#include "hw/pci/pci_device.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
