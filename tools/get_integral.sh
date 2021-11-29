
npoints=$4
ini=$2
fin=$3

step=`awk -v ini=$ini -v fin=$fin -v npoints=$npoints 'BEGIN{print (fin-ini)/npoints}'`

fgrep -v \# $1 | awk '{if(NF>0) print $0}' | awk -v step=$step 'BEGIN{f[1]=0}{i++;weight[i]=$3;coord[i]=$1;av[i]=$2;f[i]=0;if(i>1&&(weight[i]>0||weight[i-1]>0)) f[i]=f[i-1]+step*(av[i-1]*weight[i-1]+av[i]*weight[i])/(weight[i-1]+weight[i])}END{for(j=1;j<=i;j++) print coord[j],(f[j])}' > fes_analytic.out

fgrep -v \# $1 | awk '{if(NF>0) print $0}' > pippo

reff=`paste fes_analytic.out pippo | awk -v ini=$ini -v fin=$fin  '{if($NF>0&&$1>ini&&$1<fin) printf"%f %f \n",$1,$2}' | sort -n -k2 | head -n1 |  awk '{print $2}'`

paste fes_analytic.out pippo | awk -v ref=$reff 'BEGIN{f[1]=0;i=0}{i++;weight[i]=$NF;if(i==1&&weight[i]>0) print $1,$2-ref;if(i==1&&weight[i]==0) print $1,$2;if(i>1&&(weight[i]>0||weight[i-1]>0)) print $1,$2-ref;if(i>1&&(weight[i]==0&&weight[i-1]==0)) print $1,$2}' > pluto ; mv pluto fes_analytic.out

rm pippo


