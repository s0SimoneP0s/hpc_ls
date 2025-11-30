#!/bin/bash
# Script perf to CSV
# Uso: ./parse_perf.sh > output.csv

source ../../../utils/plotter_utils.sh



declare -a test_size_list=("test_mini" "test_small" "test_standard" "test_large" "test_extralarge")

touch jacobi-1d-imper_omp
rm jacobi-1d-imper_omp || true

for i in "${test_size_list[@]}"; do

    make "$i" > "input_${i}.txt" 2>&1

    case "$i" in
        test_mini)    n=500; tsteps=2 ;;
        test_small)   n=1000; tsteps=10 ;;
        test_standard)  n=10000; tsteps=100 ;;
        test_large) n=100000; tsteps=1000 ;;
        test_extralarge) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    process_input "input_${i}.txt" "$i" "$n" "$tsteps"

    rm jacobi-1d-imper_omp
done
rm input_*.txt