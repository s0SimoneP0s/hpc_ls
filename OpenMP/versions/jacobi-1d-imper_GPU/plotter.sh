#!/bin/bash
# Script perf to CSV
# Uso: ./parse_perf.sh > output.csv

source ../../../utils/plotter_utils.sh


declare -a test_size_list=("test_mini_G" "test_small_G" "test_standard_G" "test_large_G" "test_extralarge_G")

touch jacobi-1d-imper_omp
rm jacobi-1d-imper_omp || true

for i in "${test_size_list[@]}"; do

    make clean > /dev/null 2>&1
    make "$i" > "input_${i}.txt" 2>&1

    case "$i" in
        test_mini_G)    n=500; tsteps=2 ;;
        test_small_G)   n=1000; tsteps=10 ;;
        test_standard_G)  n=10000; tsteps=100 ;;
        test_large_G) n=100000; tsteps=1000 ;;
        test_extralarge_G) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    process_input "input_${i}.txt" "$i" "$n" "$tsteps"

    rm jacobi-1d-imper_omp
done
rm input_*.txt