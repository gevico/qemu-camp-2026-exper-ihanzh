/*
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2 or later, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef HW_G233_I2C_H
#define HW_G233_I2C_H

#include "hw/core/sysbus.h"
#include "qom/object.h"

#define TYPE_G233_I2C "G233i2cState"
OBJECT_DECLARE_SIMPLE_TYPE(G233I2CState, G233_I2C)

DeviceState *G233i2c_create(hwaddr addr, qemu_irq irq);

#endif
