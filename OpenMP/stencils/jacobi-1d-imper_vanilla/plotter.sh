#!/bin/bash
# Script cumulativo per estrarre dati da output perf e convertirli in CSV
# Uso: ./parse_perf.sh > output.csv

# Funzione per formattare numeri
format_number() {
    local num="$1"
    num="${num//./}"       # rimuove separatori di migliaia
    num="${num//,/.}"      # converte la virgola decimale in punto
    echo "$num"
}

# Lista dei test dataset da eseguire
declare -a test_size_list=("test_mini" "test_small" "test_standard" "test_large" "test_extralarge")


# Loop su tutti i dataset
for i in "${test_size_list[@]}"; do
    #echo "=== Esecuzione ${i} ===" >&2

    # Esegui make e salva l'output
    make "$i" > "input_${i}.txt" 2>&1

    # Valori noti dal Makefile (puoi adattarli se vuoi)
    case "$i" in
        test_mini)    n=500; tsteps=2 ;;
        test_small)   n=1000; tsteps=10 ;;
        test_standard)  n=10000; tsteps=100 ;;
        test_large) n=100000; tsteps=1000 ;;
        test_extralarge) n=1000000; tsteps=1000 ;;
        *) n=0; tsteps=0 ;;
    esac

    # Variabili temporanee
    threads=""
    time_elapsed=""
    insn_per_cycle=""
    branch_misses=""

    # Parsing file
    while IFS= read -r line; do
        # threads
        if [[ "$line" =~ ^===\ Test\ con\ ([0-9]+)\ thread ]]; then
            threads="${BASH_REMATCH[1]}"
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
        if [[ "$line" =~ ^[[:space:]]+([0-9]+[,.]?[0-9]+)[[:space:]]+seconds\ time\ elapsed ]]; then
            time_elapsed=$(format_number "${BASH_REMATCH[1]}")

            # Stampa riga CSV
            echo "${i},${n},${tsteps},${threads:-0},${time_elapsed:-0},${insn_per_cycle:-0},${branch_misses:-0}"

            # Reset
            time_elapsed=""
            insn_per_cycle=""
            branch_misses=""
        fi
    done < "input_${i}.txt"
done
rm input_*.txt