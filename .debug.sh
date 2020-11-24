#!/bin/bash
export PATH=$PATH:/home/tensorflow/alsp/riscv-fw-infrastructure/WD-Firmware/demo/build/toolchain/gcc/bin
riscv64-unknown-elf-gdb /home/tensorflow/alsp/swervolf-nexys-a7-tensorflow-lite-demo/tensorflow/tensorflow/lite/micro/tools/make/gen/zephyr_swervolf_x86_64/magic_wand/build/zephyr/zephyr.elf

