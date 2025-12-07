#!/usr/bin/env bash

# A simple library to compile and run a CUDA program to check if a GPU is available.

function check_gpu() {
    # Path relativo al file test_gpu.cu da dove viene chiamata la funzione
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local test_gpu_src="$script_dir/test_gpu.cu"
    local test_gpu_exe="$script_dir/test_gpu"
    
    # Compila test_gpu.cu
    nvcc "$test_gpu_src" -o "$test_gpu_exe"
    if [ $? -ne 0 ]; then
        echo "error: failed to compile test_gpu.cu" >&2
        return 1
    fi
    
    # Esegue il programma
    "$test_gpu_exe"
    local ret=$?
    
    # Cancella l'eseguibile
    rm -f "$test_gpu_exe"
    
    # Se il test fallisce, esce con errore
    if [ $ret -ne 0 ]; then
        echo "error: GPU test failed" >&2
        return 1
    fi
    
    return 0
}
    