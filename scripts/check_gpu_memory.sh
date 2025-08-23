#!/bin/bash

NODE=${1:-MV-DZ-MI250-03}


srun -N 1 --nodelist=$NODE --gres=gpu:1 ./scripts/run.sh run --checkpoint ${MODELBI
N_ROOT}/gpt-oss-7m.bin -m getp -n 20 -i data/input.txt -o data/o
utput.txt 

echo "Monitoring all GPUs on node: $NODE"

srun --nodelist=$NODE --gres=gpu:8 --pty bash -c '
while true; do
    clear
    echo "================================================================"
    echo "           GPU Memory Usage on $(hostname)"
    echo "                $(date)"
    echo "================================================================"
    echo
    
    printf "%-8s %-12s %-12s %-12s %-8s\n" "GPU ID" "Used (GB)" "Total (GB)" "Free (GB)" "Usage%"
    echo "--------+------------+------------+------------+--------"
    
    # Parse rocm-smi output
    rocm-smi --showmeminfo vram | awk "
    /GPU\[[0-9]+\].*VRAM Total Memory \(B\):/ {
        # Extract GPU ID and total memory
        match(\$0, /GPU\[([0-9]+)\]/, gpu_arr)
        gpu_id = gpu_arr[1]
        match(\$0, /([0-9]+)$/, total_arr)
        total_bytes = total_arr[1]
        
        # Read next line for used memory
        getline
        match(\$0, /([0-9]+)$/, used_arr)
        used_bytes = used_arr[1]
        
        # Convert bytes to GB
        used_gb = used_bytes / (1024*1024*1024)
        total_gb = total_bytes / (1024*1024*1024)
        free_gb = total_gb - used_gb
        usage_pct = (used_bytes / total_bytes) * 100
        
        printf \"%-8s %-12.2f %-12.2f %-12.2f %-7.1f%%\n\", 
               \"GPU \" gpu_id, used_gb, total_gb, free_gb, usage_pct
    }"
    
    echo
    echo "================================================================"
    echo "Press Ctrl+C to exit"
    sleep 3
done
'