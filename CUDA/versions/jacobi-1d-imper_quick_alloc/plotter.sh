#!/bin/bash
# Script perf to CSV
# Uso: ./parse_perf.sh > output.csv

format_number() {
    local num="$1"
    num="${num//,/\.}" # pointed decimal
    num="$(echo -n "$num" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')" # skipwtsp
    echo "$num"
}


declare -a test_size_list=("test_mini_CU_qa" "test_small_CU_qa" "test_standard_CU_qa" "test_large_CU_qa" "test_extralarge_CU_qa")


for i in "${test_size_list[@]}"; do


    make "$i" > "input_${i}.txt" 2>&1

    case "$i" in
        test_mini_CU_qa)    n=500; tsteps=2 ;;
        test_small_CU_qa)   n=1000; tsteps=10 ;;
        test_standard_CU_qa)  n=10000; tsteps=100 ;;
        test_large_CU_qa) n=100000; tsteps=1000 ;;
        test_extralarge_CU_qa) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    threads=""
    time_elapsed=""
    insn_per_cycle=""
    branch_misses=""

    while IFS= read -r line; do

        # threads
        if [[ "$line" =~ ^Threads:\ ([0-9]+) ]]; then
            threads="${BASH_REMATCH[1]}"
        fi

        # block size
        if [[ "$line" =~ ^Block\ Size:\ ([0-9]+) ]]; then
            b_size="${BASH_REMATCH[1]}"
        fi


        # Kernel execution time (only last run)
        if [[ "$line" =~ ^Kernel\ execution\ time:\ ([0-9.]+) ]]; then
            ket="${BASH_REMATCH[1]}"
        fi

        # insn per cycle
        if [[ "$line" =~ instructions.*#[[:space:]]+([0-9]+)[,.]([0-9]+)[[:space:]]+insn\ per\ cycle ]]; then
            insn_per_cycle="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
        elif [[ "$line" =~ instructions.*#[[:space:]]+([0-9]+)[[:space:]]+insn\ per\ cycle ]]; then
            insn_per_cycle="${BASH_REMATCH[1]}"
        fi

        # branch misses
        if [[ "$line" =~ ^[[:space:]]+([0-9]+)[.,]?([0-9]*)[[:space:]]+branch-misses ]]; then
            branch_misses="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
        fi

        # seconds time elapsed
        if [[ "$line" =~ ^[[:space:]]+([0-9]+[,.]?[0-9]+).*seconds\ time\ elapsed ]]; then
            time_elapsed=$(format_number "${BASH_REMATCH[1]}")


            # print csv
            echo "${i},${n},${tsteps},${threads:-0},${b_size:-0},${time_elapsed:-0},${insn_per_cycle:-0},${branch_misses:-0},${gpu_teams:-0},${gpu_threads_per_team:-0}"

            # Reset
            time_elapsed=""
            insn_per_cycle=""
            branch_misses=""
        fi
    done < "input_${i}.txt"
    rm jacobi-1d-imper_cuda build/*.o
done
rm input_*.txt