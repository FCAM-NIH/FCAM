#!/bin/bash

graf_file

python3.8 ${graf_file} -ff file_grad -units kcal -nsteps 1 -nofes -oneighf neighs.out -weth 1

