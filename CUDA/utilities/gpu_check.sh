#!/usr/bin/env bash


function check_gpu() {

    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local test_gpu_src="$script_dir/test_gpu.cu"
    local test_gpu_exe="$script_dir/test_gpu"

    /usr/local/cuda/bin/nvcc "$test_gpu_src" -o "$test_gpu_exe"
    if [ $? -ne 0 ]; then
        echo "error: failed to compile test_gpu.cu" >&2
        return 1
    fi
    

    "$test_gpu_exe"
    local ret=$?
    

    rm -f "$test_gpu_exe"
    
    if [ $ret -ne 0 ]; then
        echo "error: GPU test failed" >&2
        return 1
    fi
    
    return 0
}
    