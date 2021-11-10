python3.8 ../../../../calcf_vgauss.py -if input.dat -units kcal -colvars -temp 300 -nopgradb -oeff grad_on_eff_points.out > job&

# free energy by numerical integration

bash ../../../../tools/get_integral.sh grad_on_eff_points.out 0 30 1000

# free energy from KMC

python3.8 ../../../../graf_fes_kmc.py -ff grad_on_eff_points.out -units kcal -temp 698 -nsteps  100000000000 -ofesf fes.out > job&

# evaluate error from first and second block

python3.8 ../../../../graf_fes_kmc.py -ff grad_on_eff_points1.out -units kcal -temp 698 -nsteps 1000000000 -ofesf fes1.out

python3.8 ../../../../graf_fes_kmc.py -ff grad_on_eff_points2.out -units kcal -temp 698 -nsteps 1000000000 -ofesf fes2.out

