# Calculate minimum free energy path between bins 2618 (c7ax minimum) and 11109 (c7eq minimum) using systematic search
python3.8 ../../../../graf_fes_kmc.py -noforces -nofes -readfes -rfesfile ../results/fes.out -smfepath -sbmfepath 2618 -fbmfepath 11109 -temp 298 -units kj -pathtemp 1 -itpfile per_iter_path_c7eq-c7ax.out -pfile path_c7eq-c7ax.out
