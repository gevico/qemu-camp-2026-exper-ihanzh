#!/bin/bash

# 配置路径
KDIR="/home/han/project/linux-6.18.24"
BUSYBOX_INSTALL_DIR="/home/han/project/busybox-1.36.1/_install"
CROSS="riscv64-linux-gnu-"
DRIVER_SRC="gpgpu_driver.c"
TEST_SRC="gpgpu_driver_test.c"
DRIVER_HDR="gpgpu_driver.h"
MODULE_NAME="gpgpu_driver"

set -e

mkdir -p build
rm -f build/${MODULE_NAME}.ko
ln -sf ../${DRIVER_SRC} build/
ln -sf ../${DRIVER_HDR} build/
ln -sf ../${TEST_SRC} build/
ln -sf ../Makefile build/

echo "--- [1/3] 开始编译驱动 (使用 compiledb) ---"
cd build

# 检查是否安装了 compiledb
if command -v compiledb &> /dev/null; then
    # -n 是 dry-run，-w 是记录路径，compiledb 喜欢这种方式
    compiledb make KDIR=$KDIR ARCH=riscv CROSS_COMPILE=$CROSS
    make user-test
else
    echo "警告: 未找到 compiledb，将进行普通编译（无 JSON 生成）"
    make KDIR=$KDIR ARCH=riscv CROSS_COMPILE=$CROSS
fi

# 链接 JSON 到根目录
[ -f compile_commands.json ] && ln -sf build/compile_commands.json ../compile_commands.json

echo "--- [2/3] 拷贝驱动到 BusyBox 目录 ---"
pwd
cp -r ./ $BUSYBOX_INSTALL_DIR/build

echo "--- [3/3] 重新打包 rootfs.cpio ---"
cd $BUSYBOX_INSTALL_DIR
find . | cpio -o -H newc > ../rootfs.cpio

echo "--- 完成 ---"
