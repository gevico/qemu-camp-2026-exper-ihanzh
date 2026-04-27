#!/bin/bash

# 指向你的镜像位置
KERNEL_IMG="/home/han/project/linux-6.18.24/arch/riscv/boot/Image"
ROOTFS_CPIO="/home/han/project/busybox-1.36.1/rootfs.cpio"
QEMU_BIN="/home/han/project/qemu-camp-2026-exper-ihanzh/build/qemu-system-riscv64"

$QEMU_BIN -M virt,aia=aplic-imsic \
    -kernel $KERNEL_IMG \
    -initrd $ROOTFS_CPIO \
    -append "console=ttyS0 rdinit=/etc/init.d/rcS" \
    -device gpgpu \
    -nographic
