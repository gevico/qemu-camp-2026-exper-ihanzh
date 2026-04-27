MODULE_NAME := gpgpu_driver

KDIR ?= /home/han/project/linux-6.18.24
PWD := $(shell pwd)

ARCH ?= riscv
CROSS_COMPILE ?= riscv64-linux-gnu-

# 告诉内核编译系统：包含当前目录
ccflags-y += -I$(PWD)

obj-m := $(MODULE_NAME).o

all:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) CROSS_COMPILE=$(CROSS_COMPILE) clean

USER_TEST := gpgpu_driver_test
USER_TEST_CFLAGS := -std=gnu11 -Wall -Wextra -static
USER_CC ?= $(CROSS_COMPILE)gcc

user-test: $(USER_TEST)

$(USER_TEST): gpgpu_driver_test.c gpgpu_driver.h
	$(USER_CC) $(USER_TEST_CFLAGS) -o $@ gpgpu_driver_test.c

clean-user-test:
	rm -f $(USER_TEST)
