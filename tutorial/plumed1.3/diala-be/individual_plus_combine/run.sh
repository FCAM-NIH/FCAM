# calculate effective points

python3.8 ../../../../calcf_vgauss.py -if input.dat --justcalceffpoint > job&

# combine forces from differente trajectories

python3.8 ../../../../calcf_vgauss.py -if input_combine.dat

# calculate free energy

python3.8 ../../../../graf_fes_kmc.py -ff results/grad_on_eff_points_comb.out -units kj -temp 1298 -nsteps 10000000000 -weth 1 -ofesf fes_comb.out

