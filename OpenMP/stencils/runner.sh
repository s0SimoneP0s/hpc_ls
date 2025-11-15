#!/bin/bash

echo "# plot vanilla"
cd jacobi-1d-imper_vanilla
./plotter.sh
cd..

echo "# plot CPU"
cd jacobi-1d-imper_CPU
./plotter.sh
cd ..

echo "# plot GPU"
cd jacobi-1d-imper_GPU
./plotter.sh
