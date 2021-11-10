python3.8 ../../../calcf_vgauss.py -if input.dat -units kcal -colvars -colvarbias_column 2 -temp 298 > job&

# free energy from numerical integrator (only 1D)

bash ../../../tools/get_integral.sh results/grad_on_eff_points.out 5 40 175

# free energy from kmc

python3.8 ../../../graf_fes_kmc.py -ff grad_on_eff_points.out -units kcal -temp 2298 -nsteps 1000000000 -ofesf fes.out

# evaluate error from first and second block

python3.8 ../../../graf_fes_kmc.py -ff grad_on_eff_points1.out -units kcal -temp 2298 -nsteps 1000000000 -ofesf fes1.out

python3.8 ../../../graf_fes_kmc.py -ff grad_on_eff_points2.out -units kcal -temp 2298 -nsteps 1000000000 -ofesf fes2.out
