#!/bin/bash
# Script per estrarre dati da output perf in formato CSV
# make test > input.txt 2>&1
# Uso: ./parse_perf.sh input.txt > output.csv

# Funzione per formattare numeri
format_number() {
    local num="$1"
    # Rimuovi punti usati come separatori di migliaia
    num="${num//./}"
    # Converti virgola decimale in punto
    num="${num//,/.}"
    echo "$num"
}


declare -a test_size_list

test_size_list=(test2 test10 test100 test1000 test1000)
for i in ${test_size_list[@]} ; do 
    echo ${i}
    make ${i} > input_${i}.txt 2>&1

    # Stampa intestazione CSV
    echo "n,tsteps,threads,seconds_time_elapsed,insn_per_cycle,branch_misses"

    # Variabili per memorizzare i valori
    n=""
    tsteps=""
    threads=""
    time_elapsed=""
    insn_per_cycle=""
    branch_misses=""

    # Leggi il file riga per riga
    while IFS= read -r line; do
        # Estrai n (cerca esattamente "n = " seguito da numero)
        if [[ "$line" =~ ^n\ =\ ([0-9]+)$ ]]; then
            n="${BASH_REMATCH[1]}"
            echo $n
        fi
    
        # Estrai tsteps (cerca esattamente "tsteps = " seguito da numero)
        if [[ "$line" =~ ^tsteps\ =\ ([0-9]+)$ ]]; then
            tsteps="${BASH_REMATCH[1]}"
            echo $tsteps
        fi
        
        # Estrai threads (cerca esattamente "threads = " seguito da numero)
        if [[ "$line" =~ ^threads\ =\ ([0-9]+)$ ]]; then
            tsteps="${BASH_REMATCH[1]}"
            echo $tsteps
        fi
        

        # Estrai insn per cycle (cattura il valore DOPO #, gestendo , o . come decimali)
        if [[ "$line" =~ instructions.*#[[:space:]]+([0-9]+)[,.]([0-9]+)[[:space:]]+insn\ per\ cycle ]]; then
            # Caso con virgola o punto come separatore decimale (es: 0,84 o 0.84)
            insn_per_cycle="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
            echo $insn_per_cycle
        elif [[ "$line" =~ instructions.*#[[:space:]]+([0-9]+)[[:space:]]+insn\ per\ cycle ]]; then
            # Caso senza decimali
            insn_per_cycle="${BASH_REMATCH[1]}"
            echo $insn_per_cycle
        fi
        
        # Estrai branch-misses (gestisce sia . che , come separatori di migliaia)
        if [[ "$line" =~ ^[[:space:]]+([0-9]+)[.,]?([0-9]*)[[:space:]]+branch-misses ]]; then
            branch_misses="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
            echo $branch_misses
        fi
        
        # Estrai seconds time elapsed (deve iniziare con spazi)
        if [[ "$line" =~ ^[[:space:]]+([0-9]+[,.]?[0-9]+)[[:space:]]+seconds\ time\ elapsed ]]; then
            time_elapsed=$(format_number "${BASH_REMATCH[1]}")
            echo $time_elapsed

            # Quando troviamo time_elapsed, stampiamo la riga completa
            if [[ -n "$n" && -n "$tsteps" && -n "$threads" ]]; then
                echo "$n,$tsteps,$threads,$time_elapsed,$insn_per_cycle,$branch_misses"
                ((count++))
                insn_per_cycle=""
                branch_misses=""
            fi
        fi
    done < input_${i}.txt
done