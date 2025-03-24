#!/bin/sh
mkdir LightGBM/build/
make -C LightGBM/build/ clean
make -C LightGBM/build/ -j 32
cd LightGBM
sh ./build-python.sh install