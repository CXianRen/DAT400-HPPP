#!/bin/bash

# set -x 

# chage the SDE path to your installing path
export PATH=$PATH:~/Desktop/sde-external-9.24.0-2023-07-13-lin/

cat /sys/devices/cpu_atom/caps/pmu_name

thread_list=(4 8 16 32 64 128 256 512 1024)
problem_sizes=(5 1 10 100)

# clean all log before run
rm -rf sde_log/*

# 
funcDo(){
    thread=$1
    size=$2
    let size_byte=$size*1000000

    export "OMP_NUM_THREADS=$thread"
    gcc -O -DNTIMES=2 -DSTREAM_ARRAY_SIZE=${size_byte} -fopenmp stream.c -o stream.xM
    # run test
    sde64 -skx -i -global-region -omix ~/DAT400-HPPP/lab_3/sde_log/T_${thread}_size_${size}_sde.log -- ~/DAT400-HPPP/lab_3/stream.xM \
        | tee -a  ~/DAT400-HPPP/lab_3/sde_log/T_${thread}_size_${size}_sde.terminal
    # get generated log
    log_file_name=`ls sde_log/ | grep T_${thread}_size_${size}_sde.log`
    ./parse-sde.sh  sde_log/$log_file_name | tee -a ~/DAT400-HPPP/lab_3/sde_log/$log_file_name.parse
}

for thread in "${thread_list[@]}"; do
    for size in "${problem_sizes[@]}"; do
        echo "thread_${thread}_size_${size}M"
        funcDo $thread $size
    done
done

# for debug 
# funcDo 4 10
# funcDo 4 100


# basic step:

# export OMP NUM THREADS=4
# gcc -O -DNTIMES=2 -DSTREAM_ARRAY_SIZE=100000000 -fopenmp stream.c -o stream.100M
# cat /sys/devices/cpu_atom/caps/pmu_name
# sde64 -skx -i -global-region -omix ~/DAT400-HPPP/lab_3/sde.log -- ~/DAT400-HPPP/lab_3/stream.100M

# ./parse-sde.sh ./sde.log.17341-1973
