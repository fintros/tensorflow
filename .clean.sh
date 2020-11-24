#!/bin/bash
export SWERVOLF_ROOT=/home/tensorflow/alsp/swervolf-nexys-a7-tensorflow-lite-demo/swervolf
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=zephyr_swervolf BUILD_TYPE=release clean
