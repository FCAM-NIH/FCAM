# first get the labels; output file label.out containing the bin label for each frame

python3.8 ../../../../calcf_vgauss.py -if input.dat -label -noforce > job&

# count how many time each bin has been visited (necessary for getting the weight: eq. 13, 14 in the FCAM paper)

nbins=`fgrep -v \# ../results/eff_points.out | awk '{if(NF>0) print $0}' | wc | awk '{print $1}'`

fgrep -v \# label.out | awk '{if(NF>0) print $0}' | awk -v nbins=$nbins 'BEGIN{for(i=0;i<nbins;i++) av[i]=0}{av[$2]++}END{for(i=0;i<nbins;i++) print av[i]}' > count

# get trajectory of psi

fgrep -v \# ../meta.1.equil.colvars.traj  | awk '{if(NF>0) print $2}' > traj_psi

# clean label files to match traj_psi

fgrep -v \# label.out | awk '{if(NF>0) print $2}' > label

# paste tra_psi and label

paste traj_psi label > pippo ; mv pippo traj_psi

# clean free energy (necessary for the weght: eq. 13, 14 in the FCAM paper) to match count file

fgrep -v \# ../results/fes_analytic.out | awk 'BEGIN{i=0}{if(NF>0){print i,$2;i++}}' > fes

# paste free energy and count file

paste fes count > fes_count

# get weighted histogram along psi
# the weight is given by (eq. 13, 14 in the FCAM paper): exp(-[Free energy of the bin]/(kb*temp)))/[number of time the bin has been visited]

awk -v temp=298 -v kb=0.0019858775 -v fesfile=fes_count -v traj=traj_psi 'BEGIN{inte=2.0;ini=-180;fin=180;minndx=int(ini/inte)-1;maxndx=int(fin/inte);for(k=minndx;k<=maxndx;k++) num[k]=0;numtot=0}{if(FILENAME==fesfile){weight[$1]=(exp(-$2/(kb*temp)))/$(NF)};if(FILENAME==traj){if($1>=ini-inte/2&&$1<=fin+inte/2){if($1>=0)ndx=int($1/inte);if($1<0) ndx=int($1/inte)-1};num[ndx]+=weight[$2];numtot+=weight[$2]}}END{for(k=minndx+1;k<maxndx;k++) print inte/2+inte*k,num[k]/(numtot*inte),-(kb*temp)*log(num[k]/(numtot*inte))}' fes_count traj_psi > hist_psi.out

# hist_psi.out file contains: psi, probability, free energy

rm count label traj_psi fes fes_count
