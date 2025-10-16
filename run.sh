#!bin/sh


cd build
make -j16
cd ..
python demo.py bunny
