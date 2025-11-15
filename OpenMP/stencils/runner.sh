#!/bin/bash

# Stampa intestazione CSV una sola volta
echo "dataset,n,tsteps,threads,seconds_time_elapsed,insn_per_cycle,branch_misses"

cd jacobi-1d-imper_vanilla
./plotter.sh
cd ..

cd jacobi-1d-imper_CPU
./plotter.sh
cd ..

cd jacobi-1d-imper_GPU
./plotter.sh


