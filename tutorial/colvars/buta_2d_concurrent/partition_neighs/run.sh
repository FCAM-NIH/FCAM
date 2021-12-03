
cp /data/TMB-CSB/Marinelli/PROG/FCAM/tools/partition_neighs/do_partitions.sh .
cp /data/TMB-CSB/Marinelli/PROG/FCAM/tools/partition_neighs/job_tmp.sh .

bash do_partitions.sh ../results/eff_points.out ../results/grad_on_eff_points.out /data/TMB-CSB/Marinelli/PROG/FCAM/graf_fes_kmc.py

python3.8 ../../../../graf_fes_kmc.py -ff ../results/grad_on_eff_points.out -units kcal -temp 1298 -nsteps 10000000000 -weth 1 -rneighs -ineighf neighsc_tot.out -nopneighs -ofesf  fes.out

