python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -sbmfepath 2618 -fbmfepath 11109 -temp 400 -npaths 1000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_50.out -pfile path_c7eq_c7ax_ff_50.out -mctemp 50


for((i=50;i>10;i-=10)); do python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -rneighs -nopneighs -ineighf neighs.out -sbmfepath 2618 -fbmfepath 11109  -temp 400 -npaths 100000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_$((i-10)).out -pfile path_c7eq_c7ax_ff_$((i-10)).out -tpaths 1 -mctemp $((i-10)) -readpath -rpathfile path_c7eq_c7ax_ff_${i}.out ; done

for((i=10;i>1;i-=1)); do python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -rneighs -nopneighs -ineighf neighs.out -sbmfepath 2618 -fbmfepath 11109  -temp 400 -npaths 100000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_$((i-1)).out -pfile path_c7eq_c7ax_ff_$((i-1)).out -tpaths 1 -mctemp $((i-1)) -readpath -rpathfile path_c7eq_c7ax_ff_${i}.out ; done

python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -rneighs -nopneighs -ineighf neighs.out -sbmfepath 2618 -fbmfepath 11109  -temp 400 -npaths 100000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_0.5.out -pfile path_c7eq_c7ax_ff_0.5.out -tpaths 1 -mctemp 0.5 -readpath -rpathfile path_c7eq_c7ax_ff_1.out

python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -rneighs -nopneighs -ineighf neighs.out -sbmfepath 2618 -fbmfepath 11109  -temp 400 -npaths 100000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_0.4.out -pfile path_c7eq_c7ax_ff_0.4.out -tpaths 1 -mctemp 0.4 -readpath -rpathfile path_c7eq_c7ax_ff_0.5.out

python3.8 ../../../../../graf_fes_kmc.py -nofes -ff ../../results/grad_on_eff_points.out -mfepath -rneighs -nopneighs -ineighf neighs.out -sbmfepath 2618 -fbmfepath 11109  -temp 400 -npaths 100000 -pathtemp 1 -units kj -itpfile per_iter_path_c7eq_c7ax_ff_0.3.out -pfile path_c7eq_c7ax_ff_0.3.out -tpaths 1 -mctemp 0.3 -readpath -rpathfile path_c7eq_c7ax_ff_0.4.out

