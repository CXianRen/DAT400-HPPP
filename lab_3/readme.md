# run git submodule init
# run git submodule update

# Run ERT tasl
```
./ert --verbose=2 --run ./Config/config.lab3 | tee ../../ERT.log
```

[how to use gnuplt ](www.gnuplot.info/doc_5.0/gnuplot.pdf)

# how to show a .ps(PostScrip) file

gs Results.Local.Server/Run.001/roofline.ps



export PATH=$PATH:~/Desktop/sde-external-9.24.0-2023-07-13-lin/


export OMP NUM THREADS=4
gcc -O -DNTIMES=2 -DSTREAM_ARRAY_SIZE=100000000 -fopenmp stream.c -o stream.100M

cat /sys/devices/cpu_atom/caps/pmu_name

sde64 -skx -i -global-region -omix ~/DAT400-HPPP/lab_3/sde.log -- ~/DAT400-HPPP/lab_3/stream.100M

./parse-sde.sh ./sde.log.17341-1973


