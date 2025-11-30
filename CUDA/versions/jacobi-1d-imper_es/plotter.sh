#!/bin/bash
# Script perf to CSV
# Uso: ./parse_perf.sh > output.csv

source ../../../utils/plotter_utils.sh

declare -a test_size_list=("test_mini_CU_es" "test_small_CU_es" "test_standard_CU_es" "test_large_CU_es" "test_extralarge_CU_es")


for i in "${test_size_list[@]}"; do

    make clean > /dev/null 2>&1
    make "$i" > "input_${i}.txt" 2>&1

    case "$i" in
        test_mini_CU_es)    n=500; tsteps=2 ;;
        test_small_CU_es)   n=1000; tsteps=10 ;;
        test_standard_CU_es)  n=10000; tsteps=100 ;;
        test_large_CU_es) n=100000; tsteps=1000 ;;
        test_extralarge_CU_es) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    process_input "input_${i}.txt" "$i" "$n" "$tsteps"


done
rm input_*.txt