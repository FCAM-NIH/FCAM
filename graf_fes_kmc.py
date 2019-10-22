#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import time
start_time = time.time()

# read the grid with the gradients
tmparray = np.loadtxt("force_on_bin_points.out")
npoints=len(tmparray)
#ndim=int(len(tmparray[0])/2)
ndim=2
nearest=False
# now read the header
header=np.zeros((ndim,4))
count=0
f=open ("force_on_bin_points.out", 'r')
for line in f:
   count=count+1
   if count>1: 
     if count<=ndim+1:
       parts=line.split()
       header[count-2]=parts[1:]   
   if count>ndim+1:
     break

# assign lowerboundary, width, number of points for each variable and periodicity 

#with open("per_iteration_kmc_output.dat" , 'w') as f:
#    f.write("# Iteration, colvar, etc. \n")

lowbound=np.zeros((ndim))
width=np.zeros((ndim))
npointsv=np.zeros((ndim),dtype=np.int32)
period=np.zeros((ndim),dtype=np.int8)

lowbound=header[:,0]
width=header[:,1]
npointsv=header[:,2]
periodic=header[:,3] 
box=width*npointsv

# now find number of neighbours (useful for high dimensional sparse grid)

if nearest:
  maxneigh=2*ndim
else:
  maxneigh=np.power(3,ndim)-1
nneigh=np.zeros((npoints),dtype=np.int32)
neigh=np.ones((npoints,maxneigh),dtype=np.int32)
ncol=np.ma.size(tmparray,axis=1)
weights=np.zeros((npoints,ndim))
if ncol>2*ndim+1:
  weights=tmparray[:,2*ndim:3*ndim]
else:
  for j in range(0,ndim):
     weights[:,j]=tmparray[:,2*ndim]  
neigh=-neigh 
nsteps=20000000
temp=1298
kb=0.00831
prob=np.zeros((npoints,maxneigh))

totperiodic=np.sum(periodic)
# assign neighbours and probabilities
for i in range(0,npoints-1):
    diff=tmparray[i,0:ndim]-tmparray[i+1:npoints,0:ndim]
    if totperiodic>0:
      diff=diff/box
      diff=diff-np.rint(diff)*periodic
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

print ("got the neighbours")

# now start to reconstruct the free energy
# initialize the free energy to zero
# fes and probability is defined on the same points as the gradient

freq=np.zeros((npoints,maxneigh))
for j in range(0,maxneigh):
   freq[:,j]=np.sum(prob,axis=1)

prob=np.where(freq>0,prob/freq,0.0)
freq=freq[:,0]
print ("probabilities calculated")

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
for nn in range(0,npoints):
   if pop[nn]>0:
     names=str(tmparray[nn,0:ndim])
     print names[1:-1],-kb*temp*np.log(pop[nn]/maxpop)

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
