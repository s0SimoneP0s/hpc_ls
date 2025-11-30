#!/bin/bash

# Function to process input and extract metrics
process_input() {
    local input_file="$1"
    local i="$2"
    local n="$3"
    local tsteps="$4"

    threads=""
    time_elapsed=""
    insn_per_cycle=""
    branch_misses=""

    while IFS= read -r line; do

        # threads
        if [[ "$line" =~ ^===\ Test\ con\ ([0-9]+)\ thread ]]; then
            threads="${BASH_REMATCH[1]}"
        fi

        # gpu_teams
        if [[ "$line" =~ ^Teams:[[:space:]]+([0-9]+) ]]; then
            gpu_teams="${BASH_REMATCH[1]}"
        fi

        # gpu_threads_per_team
        if [[ "$line" =~ ^Thread\ limit:[[:space:]]+([0-9]+) ]]; then
            gpu_threads_per_team="${BASH_REMATCH[1]}"
        fi

        # cuda_threads
        if [[ "$line" =~ ^Threads:\ ([0-9]+) ]]; then
            cuda_threads="${BASH_REMATCH[1]}"
        fi

        # block size
        if [[ "$line" =~ ^Block\ Size:\ ([0-9]+) ]]; then
            b_size="${BASH_REMATCH[1]}"
        fi

        # Kernel execution time (only last run)
        if [[ "$line" =~ ^Kernel\ execution\ time:\ ([0-9.]+) ]]; then
            ket="${BASH_REMATCH[1]}"
        fi

        # SAXPY execution time (only last run)
        if [[ "$line" =~ ^SAXPY\ execution\ time:\ ([0-9.]+) ]]; then
            saxpy="${BASH_REMATCH[1]}"
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
            echo "${i},${n},${tsteps},${threads:-nan},${b_size:-nan},${time_elapsed:-nan},${insn_per_cycle:-nan},${branch_misses:-nan},${gpu_teams:-nan},${gpu_threads_per_team:-nan},${cuda_threads:-nan},${ket:-nan},${saxpy:-nan}"

            # Reset
            time_elapsed=""
            insn_per_cycle=""
            branch_misses=""
            threads=""
            b_size=""
            gpu_teams=""
            gpu_threads_per_team=""
            cuda_threads=""
            ket=""
            saxpy=""
        fi
    done < "$input_file"
}
