#!/bin/bash
# Script per estrarre dati da output perf in formato CSV
# make test > input.txt 2>&1
# Uso: ./parse_perf.sh input.txt > output.csv


make test > input.txt 2>&1
INPUT_FILE=$(cat  input.txt)

#INPUT_FILE="${1:-/dev/stdin}"

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
    # Estrai n
    if [[ "$line" =~ n\ =\ ([0-9]+) ]]; then
        n="${BASH_REMATCH[1]}"
    fi
    
    # Estrai tsteps
    if [[ "$line" =~ tsteps\ =\ ([0-9]+) ]]; then
        tsteps="${BASH_REMATCH[1]}"
    fi
    
    # Estrai threads
    if [[ "$line" =~ threads\ =\ ([0-9]+) ]]; then
        threads="${BASH_REMATCH[1]}"
    fi
    
    # Estrai insn per cycle
    if [[ "$line" =~ instructions.#[[:space:]]+([0-9]+[,.]?[0-9])[[:space:]]+insn\ per\ cycle ]]; then
        insn_per_cycle="${BASH_REMATCH[1]}"
        # Converti virgola in punto per formato standard
        insn_per_cycle="${insn_per_cycle//,/.}"
    fi
    
    # Estrai branch-misses
    if [[ "$line" =~ ^[[:space:]]+([0-9]+[.,]?[0-9]*)[[:space:]]+branch-misses ]]; then
        branch_misses="${BASH_REMATCH[1]}"
        # Rimuovi separatori di migliaia
        branch_misses="${branch_misses//./}"
        branch_misses="${branch_misses//,/}"
    fi
    
    # Estrai seconds time elapsed
    if [[ "$line" =~ ([0-9]+[,.]?[0-9]*)[[:space:]]+seconds\ time\ elapsed ]]; then
        time_elapsed="${BASH_REMATCH[1]}"
        # Converti virgola in punto
        time_elapsed="${time_elapsed//,/.}"
        
        # Quando troviamo time_elapsed, abbiamo tutti i dati per questa configurazione
        if [[ -n "$n" && -n "$tsteps" && -n "$threads" ]]; then
            echo "$n,$tsteps,$threads,$time_elapsed,$insn_per_cycle,$branch_misses"
        fi
    fi
done < "$INPUT_FILE"
