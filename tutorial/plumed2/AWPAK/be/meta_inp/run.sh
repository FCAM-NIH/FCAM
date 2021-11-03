mpirun -np 4 gmx_mpi mdrun -deffnm metad-be  -multi 2 -replex 1000 -plumed plumed -nsteps 10000000 >& job&

