#!/bin/bash

print_help=`awk -v print_help=$1 'BEGIN{if(print_help=="-h") print 1;else print 0}'`

if [ $print_help -eq 1 ]
then

echo "Numerical integrator to get the 1D free energy from the free energy gradient "
echo "" 
echo "usage: "
echo ""
echo "      bash get_integral.sh {pmf_gradient_file} [CV lower bound] [CV upper bound] [CV number of GRID points] "
echo ""
echo "for example:"
echo ""
echo "      bash get_integral.sh grad_on_eff_points.out -180 180 360"
echo ""

exit

fi

npoints=$4
ini=$2
fin=$3

step=`awk -v ini=$ini -v fin=$fin -v npoints=$npoints 'BEGIN{print (fin-ini)/npoints}'`

fgrep -v \# $1 | awk '{if(NF>0) print $0}' | awk -v step=$step 'BEGIN{f[1]=0}{i++;weight[i]=$3;coord[i]=$1;av[i]=$2;f[i]=0;if(i>1&&(weight[i]>0||weight[i-1]>0)) f[i]=f[i-1]+step*(av[i-1]*weight[i-1]+av[i]*weight[i])/(weight[i-1]+weight[i])}END{for(j=1;j<=i;j++) print coord[j],(f[j])}' > fes_analytic.out

fgrep -v \# $1 | awk '{if(NF>0) print $0}' > pippo

reff=`paste fes_analytic.out pippo | awk -v ini=$ini -v fin=$fin  '{if($NF>0&&$1>=ini&&$1<=fin) printf"%f %f \n",$1,$2}' | sort -n -k2 | head -n1 |  awk '{print $2}'`

paste fes_analytic.out pippo | awk -v ref=$reff 'BEGIN{f[1]=0;i=0}{i++;weight[i]=$NF;if(i==1&&weight[i]>0) print $1,$2-ref;if(i==1&&weight[i]==0) print $1,$2;if(i>1&&(weight[i]>0||weight[i-1]>0)) print $1,$2-ref;if(i>1&&(weight[i]==0&&weight[i-1]==0)) print $1,$2}' > pluto ; mv pluto fes_analytic.out

rm pippo


