#!/bin/bash

print_help=`awk -v print_help=$1 'BEGIN{if(print_help=="-h") print 1;else print 0}'`

if [ $print_help -eq 1 ]
then

echo "usage:"
echo
echo "bash do_partitions.sh {GRID points file} {free energy gradient file} {path of graf_fes_kmc.py} {partitions number} (slurm) ({1 if slurm done, else 0 })"
echo ""
echo "                      in case slurm is not used leave blank the last option on the right"
echo ""
echo "for example the following divides the size of the grid point file by 7. The run entails 28 (4x7) jobs based on 28 overlapping partitions:"
echo ""
echo "            bash do_partitions.sh ../results/eff_points.out ../results/grad_on_eff_points.out /data/TMB-CSB/Marinelli/PROG/FCAM/graf_fes_kmc.py 7"
echo ""
echo "or in case slurm is used (modify job_tmp.sh to inculde specific slurm specification; partitions etc.)"
echo ""
echo "            bash do_partitions.sh ../results/eff_points.out ../results/grad_on_eff_points.out /data/TMB-CSB/Marinelli/PROG/FCAM/graf_fes_kmc.py 7 slurm 0"
echo ""
echo "            wait that all slurm jobs are done, then type"
echo ""
echo "            bash do_partitions.sh ../results/eff_points.out ../results/grad_on_eff_points.out /data/TMB-CSB/Marinelli/PROG/FCAM/graf_fes_kmc.py 7 slurm 1"
echo ""

exit

fi

# inte provides the number of partitions for example if inte=7 the total number of "overlapping" partitions is 4xinte=28

eff_points=$1
grad_file=$2
grad_file_name=grad_on_eff_points.out
graf_file=$3
inte=$4
run_type=$5
slurmdone=$6
efflines=`cat $eff_points | fgrep -v \# | awk '{i++}END{print i}'`
divide8=`awk -v nlines=${efflines} -v div=$inte 'BEGIN{print int(0.5*0.25*nlines/div)}'`
divide4=$((divide8+divide8))
divide2=$((divide4+divide4))
divide=$((divide2+divide2))

runt=`awk -v run_type=$run_type 'BEGIN{if(run_type=="slurm") print 1;else print 0}'`

if [ $runt -eq 0 ]
then
slurmdone=0
fi

# get head

if [ $slurmdone -eq 0 ]
then

grep \# $grad_file > head

for((i=1;i<=$((4*inte-1));i++)); do cat $eff_points | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i '{if(NR>(i-1)*(divide)-(i-1)*(3*divide4)&&NR<=(i)*(divide)-(i-1)*(3*divide4)) print $0}' > eff_points_${i}.out ; done

i=$((4*inte))

cat $eff_points | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i '{if(NR>(i-1)*(divide)-(i-1)*(3*divide4)) print $0}' > eff_points_${i}.out

# 

efflines=`cat $grad_file | fgrep -v \# | awk '{i++}END{print i}'`
inte=$4
divide8=`awk -v nlines=${efflines} -v div=$inte 'BEGIN{print int(0.5*0.25*nlines/div)}'`
divide4=$((divide8+divide8))
divide2=$((divide4+divide4))
divide=$((divide2+divide2))

for((i=1;i<=$((4*inte-1));i++)); do cat $grad_file | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i '{if(NR>(i-1)*(divide)-(i-1)*(3*divide4)&&NR<=(i)*(divide)-(i-1)*(3*divide4)) print $0}' > ${grad_file_name}_${i} ; done

i=$((4*inte))

cat $grad_file | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i '{if(NR>(i-1)*(divide)-(i-1)*(3*divide4)) print $0}' > ${grad_file_name}_${i}

for((i=1;i<=$((4*inte));i++)); do cat head ${grad_file_name}_${i} > pippo ; mv pippo ${grad_file_name}_${i} ; done

rm head

# set jobs and send them

if [ $runt -eq 1 ]
then

  for((i=1;i<=$((4*inte));i++)); do sed "s/file_grad/${grad_file_name}_${i}/g" job_tmp.sh | sed "s/neighs.out/neighs_${i}.out/g" | awk -v gdir=$graf_file '{if($1=="graf_file") print "graf_file="gdir;else print $0}' > job_tmp_${i}.sh; sbatch job_tmp_${i}.sh   ; done
  exit

else 

for((i=1;i<=$((4*inte));i++)); do sed "s/file_grad/${grad_file_name}_${i}/g" job_tmp.sh | sed "s/neighs.out/neighs_${i}.out/g" | awk -v gdir=$graf_file '{if($1=="graf_file") print "graf_file="gdir;else print $0}' > job_tmp_${i}.sh; bash job_tmp_${i}.sh  ; done

fi

fi


# get numbering

if [ $runt -eq 0 ]
then
slurmdone=1
fi

if [ $slurmdone -eq 1 ]
then


efflines=`cat $grad_file | fgrep -v \# | awk '{i++}END{print i}'`
inte=$4
divide8=`awk -v nlines=${efflines} -v div=$inte 'BEGIN{print int(0.5*0.25*nlines/div)}'`
divide4=$((divide8+divide8))
divide2=$((divide4+divide4))
divide=$((divide2+divide2))

for((i=1;i<=$((4*inte-1));i++)); do cat $grad_file | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i 'BEGIN{k=-1}{k++;if(NR>(i-1)*(divide)-(i-1)*(3*divide4)&&NR<=(i)*(divide)-(i-1)*(3*divide4)) print k}' > binlist_${i}.out ; done

i=$((4*inte))

cat $grad_file | fgrep -v \# | awk -v divide=$divide -v divide4=$divide4 -v i=$i 'BEGIN{k=-1}{k++;if(NR>(i-1)*(divide)-(i-1)*(3*divide4)) print k}' > binlist_${i}.out

# combine neighbours

efflines=`cat $grad_file | fgrep -v \# | awk '{i++}END{print i}'`
inte=$4
cat neighs_1.out | fgrep -v \# | awk '{if(NF>0) print $0}' > pippo
divide8=`awk -v nlines=${efflines} -v div=$inte 'BEGIN{print int(0.5*0.25*nlines/div)}'`
divide4=$((divide8+divide8))
divide2=$((divide4+divide4))
divide=$((divide2+divide2))
awk -v divide8=$divide8 -v filebin=binlist_1.out -v fileneigh=pippo 'BEGIN{k=-1;j=0}{if(FILENAME==filebin){k++;bin[k]=$1}if(FILENAME==fileneigh){j++;if(j<=5*divide8){print $0}}}' binlist_1.out pippo > neighsc_1.out

for((i=2;i<=$((4*inte-1));i++))
do
cat neighs_${i}.out | fgrep -v \# | awk '{if(NF>0) print $0}' > pippo
divide8=`awk -v nlines=${efflines} -v div=$inte 'BEGIN{print int(0.5*0.25*nlines/div)}'`
divide4=$((divide8+divide8))
divide2=$((divide4+divide4))
divide=$((divide2+divide2))
awk -v divide8=$divide8 -v filebin=binlist_${i}.out -v fileneigh=pippo 'BEGIN{k=-1;j=0}{if(FILENAME==filebin){k++;bin[k]=$1}if(FILENAME==fileneigh){j++;if(j>3*divide8&&j<=5*divide8){printf"%i ",bin[$1];printf"%i ",$2;for(i=3;i<=NF;i++){if($i!=-1)printf"%i ",bin[$i];else printf"%i ",$i}print " "}}}' binlist_${i}.out pippo > neighsc_${i}.out
done


i=$((4*inte))
cat neighs_${i}.out | fgrep -v \# | awk '{if(NF>0) print $0}' > pippo
awk -v divide8=$divide8 -v filebin=binlist_${i}.out -v fileneigh=pippo 'BEGIN{k=-1;j=0}{if(FILENAME==filebin){k++;bin[k]=$1}if(FILENAME==fileneigh){j++;if(j>3*divide8){printf"%i ",bin[$1];printf"%i ",$2;for(i=3;i<=NF;i++){if($i!=-1) printf"%i ",bin[$i];else printf"%i ",$i}print" " }}}' binlist_${i}.out pippo > neighsc_${i}.out

neigh_files=`for((i=1;i<=$((4*inte));i++)); do echo neighsc_${i}.out ; done`

cat $neigh_files > neighsc_tot.out

# clean directory

for((i=1;i<=$((4*inte));i++)); do rm neighsc_${i}.out job_tmp_${i}.sh neighs_${i}.out ${grad_file_name}_${i} eff_points_${i}.out binlist_${i}.out; done 

rm pippo

fi
