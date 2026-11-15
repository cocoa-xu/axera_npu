# SPDX-License-Identifier: GPL-2.0-only
#
# The AX620E NPU driver, built out of tree against a configured kernel:
#   make KDIR=<kernel build directory> ARCH=arm CROSS_COMPILE=<prefix>
# Buildroot's kernel-module infrastructure builds it the same way.

obj-m := ax620e-npu.o

KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

modules_install:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules_install

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
