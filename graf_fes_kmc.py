#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import time
start_time = time.time()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ff", "--forcefile", \
                        help="file containing colvars, forces and weight of each point, weight can be different for each component", \
                        type=str, required=True)
    parser.add_argument("-temp", "--temp", help="Temperature (in Kelvin) of the kinetic motecarlo: larger temperature ensure assigning the population of high free energy regions", \
                        default=298,type=float, required=False)
    parser.add_argument("-kb", "--kb", help="Boltzmann factor for calculating the force constant (k) and defining free energy units. Default is 0.00831... kJ/mol", \
                        default=0.00831446261815324,type=float, required=False)
    parser.add_argument("-nsteps", "--numkmcsteps", help="number of kinetic montecarlo steps to calculate the free energy", \
                        default=20000000,type=int, required=False)
    parser.add_argument("-ofesf", "--outfesfile", \
                        help="output file containing the free energy for each point", \
                        default="fes.out",type=str, required=False)
    parser.add_argument("-notnearest","--notnearest", \
                        help="Do not consider only nearest neighbours", \
                        default=True, dest='do_nearest', action='store_false')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()

ifile=args.forcefile
temp=args.temp
kb=args.kb
nsteps=args.numkmcsteps
free_energy_file=args.outfesfile
nearest=args.do_nearest

with open(free_energy_file, 'w') as f:
    f.write("# Colvars Free Energy \n")

# read the grid with the gradients
tmparray = np.loadtxt(ifile)
npoints=len(tmparray)
# now read the header
count=0
f=open (ifile, 'r')
for line in f:
   parts=line.split()
   if len(parts)>0:
     if str(parts[0])=="#":
       count=count+1
       if count==1:
         ndim=int(parts[1])
       if count==2:
         lowbound=[float(parts[1])]
         width=[float(parts[2])]
         npointsv=[float(parts[3])]   
         period=[float(parts[4])]
       if count>2: 
         if count<=ndim+1:
           lowbound.append(float(parts[1]))
           width.append(float(parts[2]))
           npointsv.append(float(parts[3]))
           period.append(float(parts[4]))
       if count>ndim+1:
         break

# assign lowerboundary, width, number of points for each variable and periodicity 

#with open("per_iteration_kmc_output.dat" , 'w') as f:
#    f.write("# Iteration, colvar, etc. \n")

lowbound=np.array(lowbound)
width=np.array(width)
npointsv=np.array(npointsv)
period=np.array(period)
box=width*npointsv

# now find number of neighbours (useful for high dimensional sparse grid)

if nearest:
  maxneigh=2*ndim
else:
  maxneigh=np.power(3,ndim)-1

nneigh=np.zeros((npoints),dtype=np.int32)
neigh=np.ones((npoints,maxneigh),dtype=np.int32)
ncol=np.ma.size(tmparray,axis=1)
weights=np.ones((npoints,ndim))

if ncol>2*ndim+1:
  weights=tmparray[:,2*ndim:3*ndim]
else:
  if ncol>2*ndim:
    for j in range(0,ndim):
       weights[:,j]=tmparray[:,2*ndim]  
neigh=-neigh 
prob=np.zeros((npoints,maxneigh))

totperiodic=np.sum(period)
# assign neighbours and probabilities
for i in range(0,npoints-1):
    diff=tmparray[i,0:ndim]-tmparray[i+1:npoints,0:ndim]
    if totperiodic>0:
      diff=diff/box
      diff=diff-np.rint(diff)*period
      diff=diff*box
    if nearest:
      dist=np.sum(np.abs(diff)/width,axis=1)
      neighs=np.where(np.rint(dist)==1)
    else:
      absdist=np.abs(diff)/width
      maxdist=np.amax(absdist,axis=1)
      neighs=np.where(np.rint(maxdist)==1)
    whichneigh=neighs[0]+i+1
    neigh[i,nneigh[i]:nneigh[i]+len(whichneigh)]=whichneigh[:]    
    diffdist=diff[neighs[0],0:ndim]
    avergrad=tmparray[whichneigh,ndim:2*ndim]*weights[whichneigh,0:ndim]+tmparray[i,ndim:2*ndim]*weights[i,0:ndim]
    weighttot=weights[whichneigh,0:ndim]+weights[i,0:ndim]
    avergrad=np.where(weighttot>0,avergrad/weighttot,0)
    pippo=np.where(diffdist>0,1,0)
    pippo2=np.where(weighttot==0,1,0)
    pippo3=np.where(pippo*pippo2==1,0,1)
    valid=np.amin(pippo3,axis=1)
    energydiff=np.sum(avergrad*diffdist,axis=1)
    prob[i,nneigh[i]:nneigh[i]+len(whichneigh)]=np.where(valid>0,np.exp(energydiff/(2*kb*temp)),0.0)
    neigh[whichneigh[:],nneigh[whichneigh[:]]]=i
    prob[whichneigh[:],nneigh[whichneigh[:]]]=np.where(valid>0,np.exp(-energydiff/(2*kb*temp)),0.0)
    nneigh[i]=nneigh[i]+len(whichneigh)
    nneigh[whichneigh[:]]=nneigh[whichneigh[:]]+1 

print ("Got the neighbours")

# now start to reconstruct the free energy
# initialize the free energy to zero
# fes and probability is defined on the same points as the gradient

freq=np.zeros((npoints,maxneigh))
for j in range(0,maxneigh):
   freq[:,j]=np.sum(prob,axis=1)

prob=np.where(freq>0,prob/freq,0.0)
freq=freq[:,0]
print ("Transition probabilities calculated")

# now run KMC

minweights=np.amin(weights,axis=1)   
state=np.argmax(minweights)
timekmc=0
itt=0
#names=str(tmparray[state,0:ndim])
pop=np.zeros((npoints))
#print timekmc,names[1:-1]

#with open("per_iteration_kmc_output.dat", 'a') as f:
#    np.savetxt(f, tmparray[state,0:ndim], newline='')
#    f.write(b"\n")
#    f.write("%f " % (timekmc))
#    for i in range(0,ndim):
#       f.write("%f " % (tmparray[state,i]))
#    f.write("%i \n" % (state))

for nn in range(0,nsteps):
   thisp=0
   state_old=state
   pop[state]=pop[state]+(1/freq[state])
   rand=np.random.rand()

   #thisp=np.cumsum(prob[state,:])
   #states=np.where(thisp > rand)
   #state=neigh[state,states[0][0]]
   #timekmc=timekmc-np.log(rand)/freq[state]

   # apparently is faster to have an explicit loop
   for j in range(0,nneigh[state]): 
      thisp=thisp+prob[state,j]
      if thisp > rand:
        rand=np.random.rand() 
        timekmc=timekmc-np.log(rand)/freq[state]
        state=neigh[state,j]
        break

   #names=str(tmparray[state,0:ndim])
   #print timekmc,names[1:-1]

maxpop=np.amax(pop)
#for nn in range(0,npoints):
#   if pop[nn]>0:
#     names=str(tmparray[nn,0:ndim])
#     print names[1:-1],-kb*temp*np.log(pop[nn]/maxpop)

for nn in range (0,npoints):
   with open(free_energy_file, 'a') as f:
       if pop[nn]>0:
         for j in range (0,ndim):
            f.write("%s " % (tmparray[nn,j]))
         f.write("%s \n" % (-kb*temp*np.log(pop[nn]/maxpop)))


#   with open("per_iteration_kmc_output.dat", 'a') as f:
#       np.savetxt(f, tmparray[state,0:ndim], newline='')
#       f.write(b"\n")
#       f.write("%f " % (timekmc))
#       for i in range(0,ndim):
#          f.write("%f " % (tmparray[state,i]))
#       f.write("%i \n" % (state))

print("--- %s seconds ---" % (time.time() - start_time))

sys.exit()
#
