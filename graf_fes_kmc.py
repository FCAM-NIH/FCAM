#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import time
start_time = time.time()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-noforces","--nouseforces", \
                        help="do not use forces to run kmc but use free energy (read externally) instead ", \
                        default=True, dest='use_forces', action='store_false')
    parser.add_argument("-ff", "--forcefile", \
                        help="file containing colvars, forces and weight of each point, weight can be different for each component", \
                        type=str, required=False)
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
    parser.add_argument("-nofes","--nofreeenergy", \
                        help="Do not calculate free energy", \
                        default=True, dest='do_fes', action='store_false')
    parser.add_argument("-readfes","--readfreeenergy", \
                        help="read free energy from file", \
                        default=False, dest='read_fes', action='store_true')
    parser.add_argument("-rfesfile", "--readfreeenergyfile", \
                        help="file containing the free energy for each point to be read in input", \
                        default="fes.out",type=str, required=False)
    parser.add_argument("-readpath","--readminpath", \
                        help="read initial guess of the minimum free energy path if available (for further optimization)", \
                        default=False, dest='read_path', action='store_true')
    parser.add_argument("-rpathfile", "--readminpathfile", \
                        help="file containing the initial guess of the path (colvars, pathstate, free energy, -ln(prob))", \
                        default="path.out",type=str, required=False)
    parser.add_argument("-mfepath","--minfreeenergypath", \
                        help="Calculate minimum free energy path between two bins", \
                        default=False, dest='do_mfepath', action='store_true')
    parser.add_argument("-smfepath","--sysminfreeenergypath", \
                        help="Calculate minimum free energy path between two bins using systematic search", \
                        default=False, dest='do_spath', action='store_true')
    parser.add_argument("-sbmfepath", "--startbinmfepath", help="Initial bin of the minimum free energy path", \
                        default=-1,type=int, required=False)
    parser.add_argument("-fbmfepath", "--finalbinmfepath", help="Final bin of the minimum free energy path", \
                        default=-1,type=int, required=False)
    parser.add_argument("-tpaths", "--totpathsteps", help="total number number of steps for minimum free energy path calculation ( default is 1 )", \
                        default=1,type=int, required=False)
    parser.add_argument("-npaths", "--numpathsteps", help="number of iterations within a step for minimum free energy path calculation ( default is 1000 )", \
                        default=1000,type=int, required=False)
    parser.add_argument("-npatht", "--numpathtrials", help="number of false trials beafore going to next step in minimum free energy path calculation ( default is 10 )", \
                        default=10,type=int, required=False)
    parser.add_argument("-pathtemp", "--pathtemp", help="Temperature of the free energy path: minimum free energy path corresponds to low temperature ( default is 10K )", \
                        default=10.0,type=float, required=False)
    parser.add_argument("-mctemp", "--mctemp", help="Temperature factor ov the MC sampling for path search ( default is 10 )", \
                        default=10.0,type=float, required=False)
    parser.add_argument("-itpfile", "--periterpathfile", \
                        help="output file of the path search iterations containing the likelihood and the path lenght", \
                        default="per_iter_path_file.out",type=str, required=False)
    parser.add_argument("-pfile", "--pathfile", \
                        help="output file containing the minimized path", \
                        default="path_file.out",type=str, required=False)
    parser.add_argument("-refpath","--refinepath", \
                        help="refine calculated minimum free energy path between two bins", \
                        default=False, dest='ref_mfepath', action='store_true')


#per_iter_path_file
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()

use_forces=args.use_forces
ifile=args.forcefile
temp=args.temp
mctemp=args.mctemp
kb=args.kb
nsteps=args.numkmcsteps
free_energy_file=args.outfesfile
nearest=args.do_nearest
do_fes=args.do_fes
do_mfepath=args.do_mfepath
do_spath=args.do_spath
ref_mfepath=args.ref_mfepath
startbinmfepath=args.startbinmfepath-1
finalbinmfepath=args.finalbinmfepath-1
numpaths=args.numpathsteps
totpaths=args.totpathsteps
numpathtrials=args.numpathtrials
pathtemp=args.pathtemp
read_fes=args.read_fes
read_fes_file=args.readfreeenergyfile
read_path=args.read_path
read_path_file=args.readminpathfile
per_iter_path_file=args.periterpathfile
path_file=args.pathfile

# read the grid with the gradients

if use_forces==False:
  read_fes=True
  print ("NOTE: -noforces option enabled; kmc will be runned using free energy instead of the force,")
  print ("NOTE: please provide a free energy file through the option -rfesfile.")
  print ("NOTE: as the force file, the free energy file must contain a header containing the GRID parameters:")
  print ("NOTE: number of variables, lower boundary, width, number of points and periodicity for each variable.") 

if use_forces:
  forcearray = np.loadtxt(ifile)
  npoints=len(forcearray)

if read_fes:
  #do_fes=False
  fesarray = np.loadtxt(read_fes_file)
  nfespoints=len(fesarray)
  if use_forces:
    if nfespoints!=npoints:
      print ("ERROR: force file doesn't match free energy file") 
      sys.exit()
  else:
    npoints=nfespoints

if do_spath:
  do_mfepath=False

if do_mfepath or do_spath:
  if startbinmfepath<0 or finalbinmfepath<0:
    print ("ERROR: please set the starting and final bins for calculating the minimum free energy path")
    sys.exit()
  if startbinmfepath>npoints-1 or finalbinmfepath>npoints-1:
    print ("ERROR: for calculating the minimum free energy path, please set starting and final bins that are less than the total number of bins") 
    sys.exit()
  with open(per_iter_path_file, 'w') as f:
      f.write("# Like, path lenght \n")

# now read the header

headerfile=ifile

if use_forces==False:
  headerfile=read_fes_file 

count=0
f=open (headerfile, 'r')
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

if do_fes:
  with open(free_energy_file, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]))
         f.write("  %s  " % (npointsv[j]))
         f.write("  %s  \n" % period[j])

# now find number of neighbours (useful for high dimensional sparse grid)

if nearest:
  maxneigh=2*ndim
else:
  maxneigh=np.power(3,ndim)-1

mctemp=mctemp*maxneigh
nneigh=np.zeros((npoints),dtype=np.int32)
neigh=np.ones((npoints,maxneigh),dtype=np.int32)

if use_forces:
  tmparray=forcearray 
  ncol=np.ma.size(tmparray,axis=1)
  weights=np.ones((npoints,ndim))

  if ncol>2*ndim+1:
    weights=tmparray[:,2*ndim:3*ndim]
  else:
    if ncol>2*ndim:
      for j in range(0,ndim):
         weights[:,j]=tmparray[:,2*ndim]  
else:
  tmparray=fesarray

neigh=-neigh 
prob=np.zeros((npoints,maxneigh))
logprobpath=np.zeros((npoints,maxneigh))

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
    if use_forces:
      avergrad=tmparray[whichneigh,ndim:2*ndim]*weights[whichneigh,0:ndim]+tmparray[i,ndim:2*ndim]*weights[i,0:ndim]
      weighttot=weights[whichneigh,0:ndim]+weights[i,0:ndim]
      avergrad=np.where(weighttot>0,avergrad/weighttot,0)
      pippo=np.where(diffdist>0,1,0)
      pippo2=np.where(weighttot==0,1,0)
      pippo3=np.where(pippo*pippo2==1,0,1)
      valid=np.amin(pippo3,axis=1)
      energydiff=np.sum(avergrad*diffdist,axis=1)
    else:
      energydiff=tmparray[i,ndim]-tmparray[whichneigh,ndim]
      valid=np.where(np.isnan(energydiff),0,1)
    if do_mfepath or do_spath:
      logprobpath[i,nneigh[i]:nneigh[i]+len(whichneigh)]=np.where(valid>0,energydiff/(2*kb*pathtemp),np.nan)
      logprobpath[whichneigh[:],nneigh[whichneigh[:]]]=np.where(valid>0,-energydiff/(2*kb*pathtemp),np.nan) 
    prob[i,nneigh[i]:nneigh[i]+len(whichneigh)]=np.where(valid>0,np.exp(energydiff/(2*kb*temp)),0.0)
    neigh[whichneigh[:],nneigh[whichneigh[:]]]=i
    prob[whichneigh[:],nneigh[whichneigh[:]]]=np.where(valid>0,np.exp(-energydiff/(2*kb*temp)),0.0)
    nneigh[i]=nneigh[i]+len(whichneigh)
    nneigh[whichneigh[:]]=nneigh[whichneigh[:]]+1 

print ("Got the neighbours")

# now start to reconstruct the free energy
# initialize the free energy to zero
# fes and probability is defined on the same points as the gradient

if do_mfepath or do_spath:
  freqpath=np.zeros((npoints,maxneigh)) 
  for j in range(0,maxneigh):
     freqpath[:,j]=np.nansum(np.exp(logprobpath),axis=1)
  logprobpath=np.where(freqpath>0,logprobpath-np.log(freqpath),np.nan) 
freq=np.zeros((npoints,maxneigh))
for j in range(0,maxneigh):
   freq[:,j]=np.sum(prob,axis=1)

prob=np.where(freq>0,prob/freq,0.0)
freq=freq[:,0]
print ("Transition probabilities calculated")

if read_path:
  totpaths=1
  patharray=np.loadtxt(read_path_file)
  for j in range (0,len(patharray)):
     if j==0:
       minpathdef=[int(patharray[j,ndim])]
       minpathdefenerlike=[0.0]
     else:
       minpathdef.append(int(patharray[j,ndim]))
       minpathdefenerlike.append(-((patharray[j,ndim+2]/(2*kb*pathtemp))-np.log(freqpath[int(patharray[j-1,ndim]),0])))
  minpathdeflnlike=np.sum(minpathdefenerlike)
  minpath=minpathdef
  minpathlnlike=minpathdeflnlike
  minpathenerlike=minpathdefenerlike
  with open(per_iter_path_file, 'a') as f:
      f.write("%s %s %s \n" % (minpathdeflnlike,minpathdeflnlike,len(minpathdef)))

# now run KMC

#with open("per_iteration_kmc_output.dat", 'a') as f:
#    np.savetxt(f, tmparray[state,0:ndim], newline='')
#    f.write(b"\n")
#    f.write("%f " % (timekmc))
#    for i in range(0,ndim):
#       f.write("%f " % (tmparray[state,i]))
#    f.write("%i \n" % (state))

if do_fes:

  if use_forces:  
    minweights=np.amin(weights,axis=1)
    state=np.argmax(minweights)
  else:
    state=np.argmin(fesarray[:,ndim])
    
  timekmc=0
  itt=0
  #names=str(tmparray[state,0:ndim])
  pop=np.zeros((npoints))
  #print timekmc,names[1:-1]

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
         else:
           for j in range (0,ndim):
              f.write("%s " % (tmparray[nn,j]))
           f.write("%s \n" % (np.nan))

if do_mfepath:

  # First stage global search
  with open(per_iter_path_file, 'a') as f:
      f.write("# Start global search \n")

  #for nn in range (0,numpaths):
  for kk in range (0,totpaths):
     with open(per_iter_path_file, 'a') as f:
         f.write("# STEP: %s \n" % (kk))
     for nn in range (0,numpaths):
        if read_path:
          break
        state=startbinmfepath
        lnlike=0
        path=[state]
        enerlike=[0.0]
        count=0
        while state != finalbinmfepath:
             if nn>0 and count>numpathtrials:
               break
             thisp=0
             state_old=state
             rand=np.random.rand()
             for j in range(0,nneigh[state]):
                thisp=thisp+prob[state,j]
                if thisp > rand:
                  #if np.isnan(logprobpath[state,j]):
                  #  state=startbinmfepath
                  #  lnlike=0
                  #  path=[state]
                  #  break
                  #else:
                  lnlike=lnlike-logprobpath[state,j]
                  enerlike.append(-logprobpath[state,j])
                  state=neigh[state,j]
                  path.append(state)
                  if state==startbinmfepath: # went to the beginning; restart
                    count=count+1
                    lnlike=0
                    path=[state] 
                    enerlike=[0.0]
                  elif nn>0 and lnlike > minpathlnlike: # likelihood too large; restart
                    count=count+1
                    lnlike=0
                    state=startbinmfepath
                    path=[startbinmfepath]
                    enerlike=[0.0]
                  break  
        if nn>0 and count>numpathtrials:
          continue 
        if nn==0:
          minpath=path
          minpathlnlike=lnlike
          minpathenerlike=enerlike
        else:
          if lnlike<minpathlnlike:
            minpath=path
            minpathlnlike=lnlike  
            minpathenerlike=enerlike                   
        with open(per_iter_path_file, 'a') as f:
            f.write("%s %s %s \n" % (lnlike,np.sum(enerlike),len(path))) 
    
     # Second stage local search
     if kk==0 and read_path==False:
       minpathdef=minpath
       minpathdeflnlike=minpathlnlike
       minpathdefenerlike=minpathenerlike

     if read_path:
       minpath=minpathdef
       minpathlnlike=minpathdeflnlike
       minpathenerlike=minpathdefenerlike

     with open(per_iter_path_file, 'a') as f:
         f.write("# Start local search \n")
    
     for nn in range (0,numpaths):
        rand=np.random.rand()
        initstate=int(np.rint(rand*(len(minpath)-2)))
        #initstate=int(np.rint(1+rand*(len(minpath)-3)))
        if initstate<0:
          initstate=0 
        rand=np.random.rand()
        finstate=initstate+1+int(np.rint(rand*(len(minpath)-initstate-2)))
        if finstate<0:
          finstate=len(minpath)-1
        if finstate<=initstate:
          finstate=len(minpath)-1
        if finstate>len(minpath)-1:
          finstate=len(minpath)-1
        istate=minpath[initstate]
        fstate=minpath[finstate]
        state=istate
        lnlike=np.sum(minpathenerlike[0:initstate+1])
        path=minpath[0:initstate+1]
        enerlike=minpathenerlike[0:initstate+1] 
        count=0
        toend=False
        #while state != fstate or state != finalbinmfepath:
        while state != fstate:
             if count>numpathtrials:
               break
             if toend:
               break 
             thisp=0
             state_old=state
             rand=np.random.rand()
             for j in range(0,nneigh[state]):
                thisp=thisp+prob[state,j]
                if thisp > rand:
                  #if np.isnan(logprobpath[state,j]):
                  #  state=startbinmfepath
                  #  lnlike=0
                  #  path=[state]
                  #  break
                  #else:
                  lnlike=lnlike-logprobpath[state,j]
                  enerlike.append(-logprobpath[state,j])
                  state=neigh[state,j]
                  path.append(state)
                  totlike=lnlike+np.sum(minpathenerlike[finstate+1:len(minpath)])
                  a_mov=np.exp(-(totlike-minpathlnlike)/mctemp)
                  if state==finalbinmfepath and fstate!=finalbinmfepath: # went straight to end; consider this path
                    a_mov_s=np.exp(-(lnlike-minpathlnlike)/mctemp)
                    if lnlike<minpathlnlike:
                      toend=True
                      minpath=path
                      minpathlnlike=lnlike
                      minpathenerlike=enerlike
                      with open(per_iter_path_file, 'a') as f:
                          f.write("STRAIGHT TO END %s %s %s %s \n" % (lnlike,np.sum(enerlike),len(path),lnlike-minpathlnlike))
                    elif a_mov_s>np.random.random_sample():
                      toend=True
                      minpath=path
                      minpathlnlike=lnlike
                      minpathenerlike=enerlike
                      with open(per_iter_path_file, 'a') as f:
                          f.write("STRAIGHT TO END %s %s %s %s \n" % (lnlike,np.sum(enerlike),len(path),lnlike-minpathlnlike))
                    else:
                      count=count+1
                      lnlike=np.sum(minpathenerlike[0:initstate+1])
                      state=istate
                      path=minpath[0:initstate+1]
                      enerlike=minpathenerlike[0:initstate+1]
                  elif state==istate: # went to the beginning; restart
                    count=count+1 
                    lnlike=np.sum(minpathenerlike[0:initstate+1])
                    path=minpath[0:initstate+1]
                    enerlike=minpathenerlike[0:initstate+1]
                  elif state==startbinmfepath: # went to very beginning; restart
                    count=count+1
                    lnlike=np.sum(minpathenerlike[0:initstate+1])
                    state=istate
                    path=minpath[0:initstate+1]
                    enerlike=minpathenerlike[0:initstate+1]                   
                  elif totlike>minpathlnlike and a_mov<=np.random.random_sample(): # likelihood too large; restart  
                    count=count+1
                    lnlike=np.sum(minpathenerlike[0:initstate+1])
                    state=istate 
                    path=minpath[0:initstate+1]
                    enerlike=minpathenerlike[0:initstate+1]
                  break
        if count>numpathtrials:
          continue
        if toend:
          continue
        lnlike=lnlike+np.sum(minpathenerlike[finstate+1:len(minpath)]) 
        path.extend(minpath[finstate+1:len(minpath)])
        enerlike.extend(minpathenerlike[finstate+1:len(minpath)])
        a_mov=np.exp(-(lnlike-minpathlnlike)/mctemp)
        accepted=False
        if lnlike<minpathlnlike:
          minpath=path
          minpathlnlike=lnlike
          minpathenerlike=enerlike
          accepted=True 
        elif a_mov>np.random.random_sample():
          minpath=path
          minpathlnlike=lnlike
          minpathenerlike=enerlike 
          accepted=True 
        with open(per_iter_path_file, 'a') as f:
            f.write("%s %s %s %s %s \n" % (lnlike,np.sum(enerlike),len(path),lnlike-minpathlnlike,accepted))
        if minpathlnlike<minpathdeflnlike:
          minpathdef=minpath 
          minpathdeflnlike=minpathlnlike
          minpathdefenerlike=minpathenerlike
        with open(per_iter_path_file, 'a') as f:
            f.write("MINPATH: %s %s %s %s \n" % (minpathdeflnlike,np.sum(minpathdefenerlike),len(minpathdef),minpathlnlike-minpathdeflnlike))   
  
  with open(path_file, 'w') as f:
      f.write("# colvars, pathstate, free energy, -ln(prob) \n")
      for nn in range (0,len(minpathdef)):
         for j in range (0,ndim):
            f.write("%s " % (tmparray[minpathdef[nn],j]))
         if nn==0:
           f.write("%i %s %s \n" % (minpathdef[nn],fesarray[minpathdef[nn],ndim],minpathdefenerlike[nn]))
         else:
           f.write("%i %s %s \n" % (minpathdef[nn],fesarray[minpathdef[nn],ndim],(2*kb*pathtemp)*(-minpathdefenerlike[nn]+np.log(freqpath[minpathdef[nn-1],0]))))

if do_spath:
  #atnum=np.zeros((npoints+1),dtype=np.int32)
  e_lnprob=np.zeros((npoints))
  come_from=np.zeros((npoints),dtype=np.int32)
  e_lnprob[:]=2.0
  maxprob = 0.0
  ngrp = 1
  atnum = [startbinmfepath]
  fromwho = -1
 
  e_lnprob[startbinmfepath] = 0.0
  e_lnprob[finalbinmfepath] = 2.0
  is_end=False
  while is_end==False:
     tmp_max_lnp = 0.0
     this_max = 0
     init=True
     for i in range (0,ngrp):
        this = atnum[i]
        for j in range (0,nneigh[this]):
           this1 = neigh[this,j]
           if e_lnprob[this1]>1.5:
             this_lnprob = e_lnprob[this]+logprobpath[this,j]
             if this_lnprob>tmp_max_lnp or init:
               tmp_max_lnp = this_lnprob
               this_max = this1
               fromwho = this
               init=False
     #with open(per_iter_path_file, 'a') as f:
     #    f.write("WHERE: %s %s %s %s \n" % (ngrp,this_max,fromwho,tmp_max_lnp))
     atnum.append(this_max)
     e_lnprob[this_max] = tmp_max_lnp
     come_from[this_max] = fromwho
     ngrp = ngrp + 1     
     if this_max==finalbinmfepath:
       is_end=True 
       this1=finalbinmfepath
       this=this1
       with open(path_file, 'w') as f:
           f.write("# colvars, pathstate, free energy, -ln(prob) \n")
           for j in range (0,ndim):
              f.write("%s " % (tmparray[this,j]))
           f.write("%i %s %s \n" % (this,fesarray[this,ndim],e_lnprob[this]))
           while this1!=startbinmfepath:
                this=come_from[this1]
                for j in range (0,ndim):
                   f.write("%s " % (tmparray[this,j]))
                f.write("%i %s %s \n" % (this,fesarray[this,ndim],e_lnprob[this]))
                this1=this
           #for j in range (0,ndim):
           #   f.write("%s " % (tmparray[this,j]))
           #   f.write("%i %s %s \n" % (this,fesarray[this,ndim],e_lnprob[this])) 
       break

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
