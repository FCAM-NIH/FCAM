#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy

# read the grid with the gradients
tmparray = np.loadtxt("force_on_bin_points.out")
npoints=len(tmparray)
#ndim=int(len(tmparray[0])/2)
ndim=2

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

with open("per_iteration_kmc_output.dat" , 'w') as f:
    f.write("# Iteration, colvar, etc. \n")

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

maxneigh=2*ndim
nneigh=np.zeros((npoints),dtype=np.int32)
neigh=np.ones((npoints,maxneigh),dtype=np.int32) # not needed for now
neigh=-neigh # not needed for now
nsteps=20000000
temp=1298
kb=0.00831

grad_mp=np.zeros((2,npoints,ndim))
avail_mp=np.zeros((2,npoints,ndim),dtype=np.int8)
neigh_mp=np.zeros((2,npoints,ndim),dtype=np.int32)
grad2diff=np.zeros((npoints,ndim))
gradpdiff=np.zeros((npoints,ndim))
gradmdiff=np.zeros((npoints,ndim))
gradmtot=np.zeros((npoints,ndim))
weight_mp=np.zeros((2,npoints,ndim))
whichdim=np.ones((npoints,2*ndim),dtype=np.int32)
whichsign=np.ones((npoints,2*ndim),dtype=np.int8)
whichdim=-whichdim
whichsign=-whichsign

totperiodic=np.sum(periodic)
for i in range(0,npoints):
    diff=tmparray[i,0:ndim]-tmparray[:,0:ndim]
    if totperiodic>0:
      diff=diff/box
      diff=diff-np.rint(diff)*periodic
      diff=diff*box
    dist=np.sum(np.abs(diff)/width,axis=1)
    neighs=np.where(np.rint(dist)==1)
    whichneigh=neighs[0]
    nneigh[i]=len(whichneigh)

    # assign neighbours as well as foward and backward gradients
    for j in range(0,nneigh[i]):
        diff=tmparray[i,0:ndim]-tmparray[whichneigh[j],0:ndim]
        if totperiodic>0:
          diff=diff/box
          diff=diff-np.rint(diff)*periodic
          diff=diff*box        
        signdist=(diff)/width
        indexing_m=np.where(np.rint(signdist)==1)
        indexing_p=np.where(np.rint(signdist)==-1)      
        if indexing_m[0].size>0:
          whichdim[i,j]=indexing_m[0][0]
          whichsign[i,j]=0
        else:
          whichdim[i,j]=indexing_p[0][0]  
          whichsign[i,j]=1
        #whichsign[i,j]=0*indexing_m[0].size+1*indexing_p[0].size
        # assign availability of minus one and plus one gradient
        avail_mp[whichsign[i,j],i,whichdim[i,j]]=1
        grad_mp[whichsign[i,j],i,whichdim[i,j]]=tmparray[whichneigh[j],ndim+whichdim[i,j]]
        weight_mp[whichsign[i,j],i,whichdim[i,j]]=tmparray[whichneigh[j],2*ndim]
        neigh_mp[whichsign[i,j],i,whichdim[i,j]]=whichneigh[j]
        neigh[i,j]=whichneigh[j]

print ("got the neighbours")

#avail_mp=np.floor((avail_m+avail_p)/2)  

#avail_bmp=avail_mp[0]*avail_mp[1]

# now start to reconstruct the free energy
# initialize the free energy to zero
# fes and probability is defined on the same points as the gradient

freq=np.zeros((npoints))
prob_mp=np.zeros((2,npoints,ndim))

for j in range(0,ndim):
   energydiff=np.where(tmparray[:,2*ndim]+weight_mp[0,:,j]>0,width[j]*(tmparray[:,ndim+j]*tmparray[:,2*ndim]+grad_mp[0,:,j]*weight_mp[0,:,j])/(tmparray[:,2*ndim]+weight_mp[0,:,j]),0.0)
   prob_mp[0,:,j]=np.where(avail_mp[0,:,j]*(tmparray[:,2*ndim]+weight_mp[0,:,j])>0,np.exp(energydiff[:]/(2*kb*temp)),0.0)
   #prob_mp[0,:,j]=np.where(avail_mp[0,:,j]>0.5,1.0,0.0) # DEBUG  
   energydiff=np.where(tmparray[:,2*ndim]+weight_mp[1,:,j]>0,-width[j]*(tmparray[:,ndim+j]*tmparray[:,2*ndim]+grad_mp[1,:,j]*weight_mp[1,:,j])/(tmparray[:,2*ndim]+weight_mp[1,:,j]),0.0)
   prob_mp[1,:,j]=np.where(avail_mp[1,:,j]*(tmparray[:,2*ndim]+weight_mp[1,:,j])>0,np.exp(energydiff[:]/(2*kb*temp)),0.0)
   #prob_mp[1,:,j]=np.where(avail_mp[1,:,j]>0.5,1.0,0.0) # DEBUG

   
for j in range(0,ndim):
   freq[:]=freq[:]+prob_mp[0,:,j]+prob_mp[1,:,j]

for j in range(0,ndim):
   prob_mp[0,:,j]=np.where(freq[:]>0,prob_mp[0,:,j]/freq[:],0.0)
   prob_mp[1,:,j]=np.where(freq[:]>0,prob_mp[1,:,j]/freq[:],0.0) 

print ("probabilities calculated")

# now run KMC
   
state=np.argmax(tmparray[:,2*ndim])
time=0
itt=0
#names=str(tmparray[state,0:ndim])
pop=np.zeros((npoints))
#print time,names[1:-1]

#with open("per_iteration_kmc_output.dat", 'a') as f:
#    np.savetxt(f, tmparray[state,0:ndim], newline='')
#    f.write(b"\n")
#    f.write("%f " % (time))
#    for i in range(0,ndim):
#       f.write("%f " % (tmparray[state,i]))
#    f.write("%i \n" % (state))

for nn in range(0,nsteps):
   thisp=0
   state_old=state
   pop[state]=pop[state]+(1/freq[state])
   rand=np.random.rand()
   #sign=whichsign[state,0:nneigh[state]]
   #dim=whichdim[state,0:nneigh[state]]
   #thisp=np.cumsum(prob_mp[sign[:],state,dim[:]])
   #states=np.where(thisp > rand)
   #state=neigh[state,states[0][0]]
   #time=time-np.log(rand)/freq[state]
   for j in range(0,nneigh[state]): 
      sign=whichsign[state,j]
      dim=whichdim[state,j]
      thisp=thisp+prob_mp[sign,state,dim]
      if thisp > rand:
        rand=np.random.rand() 
        time=time-np.log(rand)/freq[state]
        state=neigh[state,j]
        break
   #names=str(tmparray[state,0:ndim])
   #print time,names[1:-1]

maxpop=np.amax(pop)
for nn in range(0,npoints):
   if pop[nn]>0:
     names=str(tmparray[nn,0:ndim])
     print names[1:-1],-kb*temp*np.log(pop[nn]/maxpop)

#   with open("per_iteration_kmc_output.dat", 'a') as f:
#       np.savetxt(f, tmparray[state,0:ndim], newline='')
#       f.write(b"\n")
#       f.write("%f " % (time))
#       for i in range(0,ndim):
#          f.write("%f " % (tmparray[state,i]))
#       f.write("%i \n" % (state))

sys.exit()
#
