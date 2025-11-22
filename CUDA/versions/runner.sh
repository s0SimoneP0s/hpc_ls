#!/bin/bash

# Stampa intestazione CSV una sola volta
echo "dataset,n,tsteps,cpu_threads,seconds_time_elapsed,insn_per_cycle,branch_misses,gpu_teams,gpu_threads_per_team"

cd jacobi-1d-imper_es
./plotter.sh
cd ..

