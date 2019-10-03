#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy

# read the grid with the gradients
tmparray = np.loadtxt("force_on_eff_points_comb.out")
npoints=len(tmparray)
#ndim=int(len(tmparray[0])/2)
ndim=2

# now read the header
header=np.zeros((ndim,4))
count=0
f=open ("force_on_eff_points_comb.out", 'r')
for line in f:
   count=count+1
   if count>1: 
     if count<=ndim+1:
       parts=line.split()
       header[count-2]=parts[1:]   
   if count>ndim+1:
     break

# assign lowerboundary, width, number of points for each variable and periodicity 

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

maxneigh=2**ndim
nneigh=np.zeros((npoints),dtype=np.int32)
#neigh=np.ones((npoints,maxneigh),dtype=np.int32) # not needed for now
#neigh=-neigh # not needed for now
tol=10**-20
nsteps=1000000
temp=300
kb=0.00831
maxvalfes=40*kb*temp
ratef=0.01
gamma=10000
rfact=1.0001
lambdamod=0
deltalambdamod=0

with open("per_iteration_output.dat" , 'w') as f:
    f.write("# Iteration, chisquare, ratef \n")


grad_m=np.zeros((npoints,ndim))
avail_m=np.zeros((npoints,ndim),dtype=np.int32)
grad_p=np.zeros((npoints,ndim))
avail_p=np.zeros((npoints,ndim),dtype=np.int32)
neigh_m=np.zeros((npoints,ndim),dtype=np.int32)
neigh_p=np.zeros((npoints,ndim),dtype=np.int32) 
grad2diff=np.zeros((npoints,ndim))
gradpdiff=np.zeros((npoints,ndim))
gradmdiff=np.zeros((npoints,ndim))
gradmtot=np.zeros((npoints,ndim))
weight_p=np.zeros((npoints,ndim))
weight_m=np.zeros((npoints,ndim))

# enforce constant weight
#tmparray[:,2*ndim]=1
#

for i in range(0,npoints):
    diff=tmparray[i,0:ndim]-tmparray[:,0:ndim]
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
        diff=diff/box
        diff=diff-np.rint(diff)*periodic
        diff=diff*box        
        signdist=(diff)/width
        indexing_m=np.where(np.rint(signdist)==1,1,0)
        # assign availability of minus one gradient
        avail_m[i,:]=avail_m[i,:]+indexing_m 
        grad_m[i,:]=grad_m[i,:]+indexing_m*tmparray[whichneigh[j],ndim:2*ndim]
        weight_m[i,:]=weight_m[i,:]+indexing_m*tmparray[whichneigh[j],2*ndim:3*ndim]
        neigh_m[i,:]=neigh_m[i,:]+whichneigh[j]*indexing_m
        indexing_p=np.where(np.rint(signdist)==-1,1,0)
        # assign availability of plus one gradient
        avail_p[i,:]=avail_p[i,:]+indexing_p
        grad_p[i,:]=grad_p[i,:]+indexing_p*tmparray[whichneigh[j],ndim:2*ndim]
        weight_p[i,:]=weight_p[i,:]+indexing_p*tmparray[whichneigh[j],2*ndim:3*ndim]
        neigh_p[i,:]=neigh_p[i,:]+whichneigh[j]*indexing_p
        #neigh[i,j]=whichneigh[j] #not needed for now 

#avail_mp=np.floor((avail_m+avail_p)/2)  

avail_mp=avail_m*avail_p

# now start to reconstruct the free energy
# initialize the free energy to zero
# fes and probability is defined on the same points as the gradient

fes=np.zeros((npoints)) # start from a uniform distribution
gradfes=np.zeros((npoints,ndim))
grad2fes=np.zeros((npoints,ndim))
gradpfes=np.zeros((npoints,ndim))
gradmfes=np.zeros((npoints,ndim))

gradfes_m=np.zeros((npoints,ndim))
gradfes_p=np.zeros((npoints,ndim))
gradfw=np.zeros((npoints,ndim))
gradfw_m=np.zeros((npoints,ndim))
gradfw_p=np.zeros((npoints,ndim))
weight=np.zeros((npoints,ndim))
#weight_p=np.zeros((npoints,ndim))
#weight_m=np.zeros((npoints,ndim))

#weight_m=np.where(avail_m<0.5,0.1,1) # check
#weight_p=np.where(avail_p<0.5,0.1,1) # check
weight=np.where(tmparray[:,ndim:2*ndim]==0,0.1,1) # check

fesp=np.zeros((npoints,ndim))
fesm=np.zeros((npoints,ndim))

for j in range(0,ndim):
   gradfw[:,j]=np.where(avail_mp[:,j]>0.5,0.5/width[j],1/width[j])
   weight_p[:,j]=weight_p[:,j]/np.sum(weight_p[:,j]) 
   weight_m[:,j]=weight_m[:,j]/np.sum(weight_m[:,j])
#   gradfw[:,j]=gradfw[:,j]*((avail_mp[:,j]*(1/(2*width[j])))+((1-avail_mp[:,j])*(1/width[j])))
   
for j in range(0,ndim):
   #weight_m[:,j]=weight[neigh_m[:,j],j]
   #weight_p[:,j]=weight[neigh_p[:,j],j]
   weight_m[:,j]=weight_m[:,j]*weight[neigh_m[:,j],j]
   weight_p[:,j]=weight_p[:,j]*weight[neigh_p[:,j],j]


#   gradfw_m[:,j]=gradfw_m[:,j]*gradfw[neigh_m[:,j],j] # check
#   gradfw_p[:,j]=gradfw_p[:,j]*gradfw[neigh_p[:,j],j] # check
   gradfw_m[:,j]=gradfw[neigh_m[:,j],j]
   gradfw_p[:,j]=gradfw[neigh_p[:,j],j]

chisquareold=0
chisquare=0
for n in range(0,nsteps):

    #prob=np.exp(-fes/(kb*temp))/(np.sum(np.exp(-fes/(kb*temp))))

    # calculate FES gradients

    for j in range(0,ndim):
       fesp[:,j]=np.where(avail_p[:,j]>0.5,fes[neigh_p[:,j]],fes[:])
       fesm[:,j]=np.where(avail_m[:,j]>0.5,fes[neigh_m[:,j]],fes[:])
       gradfes[:,j]=gradfw[:,j]*(fesp[:,j]-fesm[:,j]) # finite diff 
       #fesp[:,j]=avail_p[:,j]*fes[neigh_p[:,j]]+(1-avail_p[:,j])*fes[:]
       #fesm[:,j]=avail_m[:,j]*fes[neigh_m[:,j]]+(1-avail_p[:,j])*fes[:] 
       #gradpfes[:,j]=(fesp[:,j]-fes[:])/width[j] # forward finite diff used only at the border
       #gradmfes[:,j]=(fes[:]-fesm[:,j])/width[j] # backward finite diff used only at the border

       #grad2fes[:,j]=(fes[neigh_p[:,j]]-fes[neigh_m[:,j]])/(2*width[j]) # central finite diff used in central points
       #gradfes[:,j]=(fesp[:,j]-fesm[:,j])/(2*width[j]) # finite diff
       #gradpfes[:,j]=(fes[neigh_p[:,j]]-fes[:])/(width[j]) # forward finite diff used only at the border
       #gradmfes[:,j]=(fes[:]-fes[neigh_m[:,j]])/(width[j]) # backward finite diff used only at the border
    #for j in range(0,ndim):
    #   gradfes[:,j]=(avail_mp[:,j]*grad2fes[:,j])+((1-avail_mp[:,j])*(avail_p[:,j]*gradpfes[:,j]))+((1-avail_mp[:,j])*(avail_m[:,j]*gradmfes[:,j]))

    #gradfes[:,0:ndim]=(avail_mp[:,0:ndim]*grad2fes[:,0:ndim])

    for j in range(0,ndim):
       gradfes_m[:,j]=np.where(avail_m[:,j]>0.5,gradfes[neigh_m[:,j],j],0.0)
       gradfes_p[:,j]=np.where(avail_p[:,j]>0.5,gradfes[neigh_p[:,j],j],0.0)
         
    #chisquare=(np.sum(gradfw[:,0:ndim]*(gradfes[:,0:ndim]-tmparray[:,ndim:])*(gradfes[:,0:ndim]-tmparray[:,ndim:])))/(npoints*ndim)

    chisquareold=chisquare
    chisquare=(np.sum(weight[:,0:ndim]*(gradfes[:,0:ndim]-tmparray[:,ndim:2*ndim])*(gradfes[:,0:ndim]-tmparray[:,ndim:2*ndim])))/(npoints*ndim)

    grad2diff=(weight_m[:,0:ndim]*(gradfes_m[:,0:ndim]-grad_m[:,0:ndim])-weight_p[:,0:ndim]*(gradfes_p[:,0:ndim]-grad_p[:,0:ndim]))/(2*width)
    #gradpdiff=((gradfes[:,0:ndim]-tmparray[:,ndim:])/width)-((gradfes_p[:,0:ndim]-grad_p[:,0:ndim])/(2*width))
    #gradmdiff=((gradfes_m[:,0:ndim]-grad_m[:,0:ndim])/(2*width))-((gradfes[:,0:ndim]-tmparray[:,ndim:])/width)

    #grad2diff=(gradfw_m[:,0:ndim]*(gradfes_m[:,0:ndim]-grad_m[:,0:ndim])-gradfw_p[:,0:ndim]*(gradfes_p[:,0:ndim]-grad_p[:,0:ndim]))
    #for j in range(0,ndim):
    #   grad2diff[:,j]=np.where(avail_mp[:,j]>0.5,(gradfw_m[:,j]*(gradfes_m[:,j]-grad_m[:,j])-gradfw_p[:,j]*(gradfes_p[:,j]-grad_p[:,j])),0.0)
    #   gradpdiff[:,j]=np.where(avail_mp[:,j]<0.5,avail_p[:,j]*((gradfw[:,j]*(gradfes[:,j]-tmparray[:,j+ndim]))-(gradfw_p[:,j]*(gradfes_p[:,j]-grad_p[:,j]))),0.0)
    #   gradmdiff[:,j]=np.where(avail_mp[:,j]<0.5,avail_m[:,j]*((gradfw_m[:,j]*(gradfes_m[:,j]-grad_m[:,j]))-(gradfw[:,j]*(gradfes[:,j]-tmparray[:,j+ndim]))),0.0)
    #gradtot=grad2diff+gradpdiff+gradmdiff
    #sumgradients=np.sum((gradtot[:,0:ndim]),axis=1)
    #sumgradients=np.sum((avail_mp[:,0:ndim]*grad2diff[:,0:ndim])+((1-avail_mp[:,0:ndim])*(avail_p[:,0:ndim]*gradpdiff[:,0:ndim]))+((1-avail_mp[:,0:ndim])*(avail_m[:,0:ndim]*gradmdiff[:,0:ndim])),axis=1)
    sumgradients=np.sum((grad2diff[:,0:ndim]),axis=1)
    #targetval=-(kb*temp*gamma/prob)*(sumgradients)
    targetval=-(kb*temp*gamma)*(sumgradients)
    rate=ratef/(np.mean(np.abs(targetval)))
    fesold=fes 
    fesnew=(1-rate)*fesold+rate*kb*temp*targetval
    fesnew=fesnew-np.mean(fesnew) 
    fes=np.where(np.abs(fesnew)<maxvalfes,fesnew,fesold)    
    fes=np.where(nneigh>0,fes,fesold)

    # useful to define stop criteria
    lambdamodold=lambdamod
    lambdamod=np.sum(np.abs(fes))

    deltalambdamodold=deltalambdamod
    deltalambdamod=(lambdamod-lambdamodold)/lambdamod

    if nsteps>100: # Only start updating the step size (rate) after the first 100 steps
       if deltalambdamod*deltalambdamodold<0:
          ratef=ratef/rfact
 
    if nsteps>100:
       if chisquare>chisquareold:
          ratef=ratef/rfact

    if np.abs(deltalambdamod)<tol:
       break

    if lambdamod<tol:
       break

    with open("per_iteration_output.dat", 'a') as f:
        f.write("%s %s %s %s \n" % (n,chisquare,ratef,lambdamod))

np.savetxt("final_coord.dat",(tmparray[:,0:ndim]))
np.savetxt("final_fes.dat",(fes))
with open("per_iteration_output.dat", 'a') as f:
    f.write("%s %s %s \n" % (n,chisquare,ratef))

sys.exit()

