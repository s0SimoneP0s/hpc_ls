#!/bin/bash
# Script perf to CSV
# Uso: ./parse_perf.sh > output.csv

source ../../../utils/plotter_utils.sh

declare -a test_size_list=("test_mini_CU_qa" "test_small_CU_qa" "test_standard_CU_qa" "test_large_CU_qa" "test_extralarge_CU_qa")

touch jacobi-1d-imper_cuda
rm jacobi-1d-imper_cuda build/*.o || true

for i in "${test_size_list[@]}"; do

    make clean > /dev/null 2>&1
    make "$i" > "input_${i}.txt" 2>&1

    case "$i" in
        test_mini_CU_qa)    n=500; tsteps=2 ;;
        test_small_CU_qa)   n=1000; tsteps=10 ;;
        test_standard_CU_qa)  n=10000; tsteps=100 ;;
        test_large_CU_qa) n=100000; tsteps=1000 ;;
        test_extralarge_CU_qa) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    process_input "input_${i}.txt" "$i" "$n" "$tsteps"

    rm jacobi-1d-imper_cuda build/*.o
done
rm input_*.txt