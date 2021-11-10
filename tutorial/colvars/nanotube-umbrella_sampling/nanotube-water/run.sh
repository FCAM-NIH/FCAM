python3.8 ../../../../../calcf_vgauss.py -if input.dat -units kcal -colvars -temp 300 -nopgradb -oeff grad_on_eff_points_400.out > job&

# free energy with numerical integrator (only 1D)

bash ../../../../tools/get_integral.sh grad_on_eff_points_400.out 0 7 400

# free energy with kmc

python3.8 ../../../../../graf_fes_kmc.py -ff grad_on_eff_points_400.out -units kcal -temp 698 -nsteps 1000000000 -ofesf fes_400.out

# evaluate error from first and second block

python3.8 ../../../../../graf_fes_kmc.py -ff grad_on_eff_points1.out -units kcal -temp 698 -nsteps 1000000000 -ofesf fes1_400.out

python3.8 ../../../../../graf_fes_kmc.py -ff grad_on_eff_points2.out -units kcal -temp 698 -nsteps 1000000000 -ofesf fes2_400.out

