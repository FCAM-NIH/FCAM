#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
from numba import jit
import time
start_time = time.time()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-noforces","--nouseforces", \
                        help="do not use forces to run kmc but use free energy (read externally) instead ", \
                        default=True, dest='use_forces', action='store_false')
    parser.add_argument("-rfd","--reversefinitediff", \
                        help="Calculate free energy using iterative reverse finite difference over a KMC trajectory (instead of from KMC populations)", \
                        default=False, dest='do_rfd', action='store_true')
    parser.add_argument("-kcont","--dokineticcontacts", \
                        help="Calculate neighbours from inter-bin transitions evaluated on a continous trajectory with bin-labels", \
                        default=False, dest='do_kcont', action='store_true')
    parser.add_argument("-labelsf", "--labelsfile", \
                        help="input file containing time, colvars and bin-label across a continuous trajectory (relevant with the option -kcont) ", \
			default="labels.out",type=str, required=False)
    parser.add_argument("-minbintrans", "--minnumbintransitions", help="Minimum number of transitions between two bins to be considered neighbors (relevant with the option -kcont)", \
                        default=0,type=int, required=False)
    parser.add_argument("-rneighs","--readneighs", \
                        help="read neighbours from file ", \
                        default=False, dest='read_neighs', action='store_true')
    parser.add_argument("-ineighf", "--inneighfile", \
                        help="input file containing the neighbours of each bin", \
                        default="neighs.out",type=str, required=False)
    parser.add_argument("-nopneighs","--noprintneighs", \
                        help="do not print list of bins and corresponding neighbours ", \
                        default=True, dest='print_neighs', action='store_false')
    parser.add_argument("-oneighf", "--outneighfile", \
                        help="output file containing the neighbours of each bin", \
                        default="neighs.out",type=str, required=False)
    parser.add_argument("-ff", "--forcefile", \
                        help="file containing colvars, forces and weight of each point, weight can be different for each component", \
                        type=str, required=False)
    parser.add_argument("-temp", "--temp", help="Temperature (in Kelvin) of the kinetic motecarlo: larger temperature ensure assigning the population of high free energy regions", \
                        default=298,type=float, required=False)
    parser.add_argument("-units", "--units", \
                        help="Choose free energy units specifying (case sensitive) either kj (kj/mol) or kcal (kcal/mol) (in alternative you can set the Boltzmann factor through the option -kb)", \
                        type=str, required=False)
    parser.add_argument("-kb", "--kb", help="Boltzmann factor for calculating the force constant (k) and defining free energy units.", \
                        default=-1,type=float, required=False)
    parser.add_argument("-nsteps", "--numkmcsteps", help="number of kinetic montecarlo steps to calculate the free energy", \
                        default=20000000,type=int, required=False)
    parser.add_argument("-weth", "--wethreshold", help="Minimum value of the smallest weight for a state to be considered ", \
                        default=-1.0,type=float, required=False)
    parser.add_argument("-maxfes", "--maxfes", help="Discard states having free energy larger than maxfes", \
                        default=-1.0,type=float, required=False)
    parser.add_argument("-minneighs", "--minneighs", help="Minimum number of neighbors for a state to be valid", \
                        default=0,type=int, required=False)
    parser.add_argument("-dexp", "--distexp", help="exponent to weight the distances in the transition probability (1/(d^dexp))", \
                        default=2.0,type=float, required=False)
    parser.add_argument("-ofesf", "--outfesfile", \
                        help="output file containing the free energy for each point", \
                        default="fes.out",type=str, required=False)
    parser.add_argument("-notnearest","--notnearest", \
                        help="Do not consider only nearest neighbours", \
                        default=True, dest='do_nearest', action='store_false')
    parser.add_argument("-cutoff","--cutoff", \
                        help="use cutoff to calculate neighbours", \
                        default=False, dest='do_cutoff', action='store_true')
    parser.add_argument("-ctval", "--cutoffval", help="value of the cutoff (in units of width) to calculate the neighbours", \
                        default=2.0,type=float, required=False)
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
    parser.add_argument("-mcpath","--montecarlopath", \
                        help="Do a montecarlo of the paths between two bins (you can minimize by gradually reducing mctemp)", \
                        default=False, dest='mc_mfepath', action='store_true')


#per_iter_path_file
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()

use_forces=args.use_forces
print_neighs=args.print_neighs
do_kinetic_cont=args.do_kcont
do_rfd=args.do_rfd
labelsfile=args.labelsfile
mintrans=args.minnumbintransitions
read_neighs=args.read_neighs
neighs_input_file=args.inneighfile
ifile=args.forcefile
temp=args.temp
mctemp=args.mctemp
units=args.units
kb=args.kb
nsteps=args.numkmcsteps
dexp=args.distexp
wethreshold=args.wethreshold
maxfes=args.maxfes
minneighs=args.minneighs
free_energy_file=args.outfesfile
neighs_file=args.outneighfile
nearest=args.do_nearest
do_fes=args.do_fes
do_mfepath=args.do_mfepath
do_spath=args.do_spath
do_cutoff=args.do_cutoff
cutoff=args.cutoffval
mc_mfepath=args.mc_mfepath
startbinmfepath=args.startbinmfepath
finalbinmfepath=args.finalbinmfepath
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

if str(units)=="kj":
  kb=0.00831446261815324
elif str(units)=="kcal":
  kb=0.0019858775
elif kb<0:
    print ("ERROR: please specify either the units (-units) or the value of the Boltzmann factor (-kb option)")
    sys.exit() 

if do_cutoff:
  nearest=False
# read the grid with the gradients

if read_neighs:
  calc_neighs=False
else:
  calc_neighs=True

if use_forces==False:
  read_fes=True
  print ("NOTE: -noforces option enabled; kmc will be runned using free energy instead of the force,")
  print ("NOTE: please provide a free energy file through the option -rfesfile.")
  print ("NOTE: as the force file, the free energy file must contain a header containing the GRID parameters:")
  print ("NOTE: number of variables, lower boundary, width, number of points and periodicity for each variable.") 

if use_forces:
  forcearray = np.loadtxt(ifile)
  npoints=len(forcearray)

if do_kinetic_cont and calc_neighs:
  print ("reading labels file")
  labelsarray = np.loadtxt(labelsfile) 
  labelspoints = len(labelsarray)

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

if print_neighs:
  with open(neighs_file, 'w') as f:
      f.write("# bin, nneighs, neighs \n")

if nearest:
  maxneigh=2*ndim
elif do_cutoff:
  maxneigh=int(npoints/2)
else:
  maxneigh=np.power(3,ndim)-1

if do_kinetic_cont:
  maxneigh=np.power(3,ndim)-1

#mctemp=mctemp*maxneigh
mctemp=mctemp*2*ndim

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

totperiodic=np.sum(period)
# assign neighbours and probabilities

# set valid states

if use_forces:
  minweight=np.amin(weights,axis=1)
  validstates=np.where(minweight>wethreshold)
  stateisvalid=np.where(minweight>wethreshold,1,0)

if read_fes:
  if maxfes>0:
    validstates=np.where(fesarray[:,ndim]<=maxfes)
    stateisvalid=np.where(fesarray[:,ndim]<=maxfes,1,0)
  else:
    validstates=np.where(np.isfinite(fesarray[:,ndim]))
    stateisvalid=np.where(np.isfinite(fesarray[:,ndim]),1,0)

def calc_neighs_fast(numpoints):
   nneighb=np.zeros((numpoints),dtype=np.int32)
   neighb=np.ones((numpoints,maxneigh),dtype=np.int32)
   neighb=-neighb

   totpoints=len(validstates[0])   
   thisarray=tmparray[validstates[0],0:ndim]
   
   for i in range(0,totpoints-1):
       diff=thisarray[i,0:ndim]-thisarray[i+1:totpoints,0:ndim]
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
         if do_cutoff:
           neighs=np.where(maxdist<=cutoff)
         else: 
           neighs=np.where(np.rint(maxdist)==1)
       whichneigh=neighs[0]+i+1
       j=validstates[0][i]
       neighb[j,nneighb[j]:nneighb[j]+len(whichneigh)]=validstates[0][whichneigh[0:len(whichneigh)]]  
       neighb[validstates[0][whichneigh[:]],nneighb[validstates[0][whichneigh[:]]]]=j 
       nneighb[j]=nneighb[j]+len(whichneigh)
       nneighb[validstates[0][whichneigh[:]]]=nneighb[validstates[0][whichneigh[:]]]+1 
   return nneighb,neighb

def calc_neighs_kcont(numpoints,mintransitions,maxn):
   nneighb=np.zeros((numpoints),dtype=np.int32)
   neighb=np.ones((numpoints,maxn),dtype=np.int32)
   numtrans=np.zeros((numpoints,maxn),dtype=np.int32) 
   #distancen=np.zeros((numpoints,maxn))
   neighb=-neighb
   ncoll=np.ma.size(labelsarray,axis=1)
   for i in range(0,labelspoints-1):
      if labelsarray[i+1,0]>=0 and labelsarray[i,0]>=0:
        step=labelsarray[i+1,0]-labelsarray[i,0]
        break
   if step<0:
    print ("ERROR: negative step in labels file")
    sys.exit()

   for i in range(0,labelspoints-1):
      time1=labelsarray[i,0]
      time2=labelsarray[i+1,0]
      bin1=int(labelsarray[i,ncoll-2])
      bin2=int(labelsarray[i+1,ncoll-2])
      if bin1>0 and bin2>0 and bin1!=bin2 and stateisvalid[bin1] and stateisvalid[bin2]:
        if time2-time1>0.1*step and time2-time1<1.1*step:
           #diff=labelsarray[i,1:ndim+1]-labelsarray[i+1,1:ndim+1]
           diff=tmparray[bin1,0:ndim]-tmparray[bin2,0:ndim]
           if totperiodic>0:
             diff=diff/box
             diff=diff-np.rint(diff)*period
             diff=diff*box
           absdist=np.abs(diff)/width
           maxdist=np.amax(absdist)
           if nneighb[bin1]>0:
             whichneighs=neighb[bin1,0:nneighb[bin1]]
             checkneigh=np.where(whichneighs==bin2) 
             if (len(checkneigh[0]))>0: 
               numtrans[bin1,checkneigh[0]]=numtrans[bin1,checkneigh[0]]+1  
               whichneighs2=neighb[bin2,0:nneighb[bin2]]
               checkneigh2=np.where(whichneighs2==bin1)
               numtrans[bin2,checkneigh2[0]]=numtrans[bin2,checkneigh2[0]]+1
             else:
               if maxdist<1.1:
                 neighb[bin1,nneighb[bin1]]=bin2
                 neighb[bin2,nneighb[bin2]]=bin1            
                 #distancen[bin1,nneighb[bin1]]=maxdist
                 #distancen[bin2,nneighb[bin2]]=maxdist
                 numtrans[bin1,nneighb[bin1]]=numtrans[bin1,nneighb[bin1]]+1
                 numtrans[bin2,nneighb[bin2]]=numtrans[bin2,nneighb[bin2]]+1
                 nneighb[bin1]=nneighb[bin1]+1
                 nneighb[bin2]=nneighb[bin2]+1 
           else:
             if maxdist<1.1: 
               neighb[bin1,nneighb[bin1]]=bin2
               neighb[bin2,nneighb[bin2]]=bin1
               #distancen[bin1,nneighb[bin1]]=maxdist
               #distancen[bin2,nneighb[bin2]]=maxdist 
               numtrans[bin1,nneighb[bin1]]=numtrans[bin1,nneighb[bin1]]+1
               numtrans[bin2,nneighb[bin2]]=numtrans[bin2,nneighb[bin2]]+1 
               nneighb[bin1]=nneighb[bin1]+1
               nneighb[bin2]=nneighb[bin2]+1
   maxn=np.amax(nneighb)
   neighb2=neighb[:,0:maxn]
#   neighb2=neighb
   nneighb2=nneighb
   numtrans2=numtrans
   #distancen2=distancen
   nneighb=np.zeros((numpoints),dtype=np.int32)
   neighb=np.ones((numpoints,maxn),dtype=np.int32)
   numtrans=np.zeros((numpoints,maxn),dtype=np.int32)
   #distancen=np.zeros((numpoints,maxn))
   neighb=-neighb
   for i in range(0,numpoints):
      if nneighb2[i]>0:
        goodtrans=np.where(numtrans2[i,0:nneighb2[i]]>mintransitions)
        nneighb[i]=len(goodtrans[0])
        if nneighb[i]>0:
          neighb[i,0:nneighb[i]]=neighb2[i,goodtrans[0]]
          numtrans[i,0:nneighb[i]]=numtrans2[i,goodtrans[0]]
          #distancen[i,0:nneighb[i]]=distancen2[i,goodtrans[0]]

   return nneighb,neighb

def check_neighs(numpoints,nneighb,neighb):
   for i in range(0,numpoints):
      whichneighs=neighb[i,0:nneighb[i]]
      for j in range(0,len(whichneighs)):
         whichneighs2=np.array(neighb[whichneighs[j],0:nneighb[whichneighs[j]]])
         checkneigh=np.where(whichneighs2==i)
         if (len(checkneigh[0]))==0:
           neighb[whichneighs[j],nneighb[whichneighs[j]]]=i
           nneighb[whichneighs[j]]=nneighb[whichneighs[j]]+1
   neighb2=neighb
   nneighb2=nneighb
   nneighb=np.zeros((numpoints),dtype=np.int32)
   neighb=np.ones((numpoints,maxneigh),dtype=np.int32)
   neighb=-neighb
   for i in range(0,numpoints):
      if nneighb2[i]<minneighs:
        stateisvalid[i]=0
   for i in range(0,numpoints):
      if stateisvalid[i] and nneighb2[i]>0:
        goodtrans=np.where(stateisvalid[neighb2[i,0:nneighb2[i]]])    
        nneighb[i]=len(goodtrans[0])
        if nneighb[i]>0:
          neighb[i,0:nneighb[i]]=neighb2[i,goodtrans[0]] 
 
   return nneighb,neighb

@jit(nopython=True)
def run_kmc(numsteps,numpoints,startstate):

   state=startstate 
   timekmc=0
   itt=0
   #names=str(tmparray[state,0:ndim])
   popu=np.zeros((numpoints))

   for nn in range(0,numsteps):
      thisp=0
      state_old=state
      popu[state]=popu[state]+(1/freq[state])
      rand=np.random.rand()
  
  
      #  thisp=np.cumsum(prob[state,0:nneigh[state]])
      #  states=np.where(thisp > rand)
      #  state=neigh[state,states[0][0]]
      #  timekmc=timekmc-np.log(rand)/freq[state]
  
      # apparently is faster to have an explicit loop
  
      #else:
      for j in range(0,nneigh[state]):
         thisp=thisp+prob[state,j]
         if thisp > rand and freq[neigh[state,j]]>0:
           rand=np.random.rand()
           timekmc=timekmc-np.log(rand)/freq[state]
           state=neigh[state,j]
           break
   return popu   

@jit(nopython=True)
def run_kmc_rfd(numsteps,numpoints,startstate):

   state=startstate
   itt=0
   #names=str(tmparray[state,0:ndim])
   fesu=np.empty((numpoints))
   fesu[:]=np.nan
   fesu[state]=0
    
   for nn in range(0,numsteps):
      thisp=0
      rand=np.random.rand()
      # assign free energy of neighbors based on reverse finite differences if unassigned
      for j in range(0,nneigh[state]):
         if np.isnan(fesu[neigh[state,j]]) and freq[neigh[state,j]]>0 and prob[state,j]>0:
           fesu[neigh[state,j]]=fesu[state]-fesdiff[state,j]
         #print ("DEBUG",state,j,neigh[state,j],fesu[neigh[state,j]],fesu[state],fesdiff[state,j],prob[state,j],freq[neigh[state,j]])
      # correct free energy of current state by weighted average of free energy differences over neighbors
      weighttotu=0
      fesave=0 
      for j in range(0,nneigh[state]):
         if freq[neigh[state,j]]>0 and prob[state,j]>0:
           weightu=weights[neigh[state,j],0:ndim]+weights[state,0:ndim]
           weighttotu=weighttotu+np.mean(weightu)
           fesave=fesave+((fesu[neigh[state,j]]+fesdiff[state,j])*np.mean(weightu))
      fesu[state]=fesave/weighttotu
      for j in range(0,nneigh[state]):
         thisp=thisp+prob[state,j]
         if thisp > rand and freq[neigh[state,j]]>0:
           rand=np.random.rand()
           state=neigh[state,j]
           break
   return fesu

if calc_neighs:  
  if do_kinetic_cont:
    print ("Calculating the neighbours using kinetic contacts")
    nneigh,neigh=calc_neighs_kcont(npoints,mintrans,maxneigh)
    maxneigh=np.amax(nneigh) 
  else:
    print ("Calculating the neighbours using geometric contacts")
    nneigh,neigh=calc_neighs_fast(npoints)
  nneigh,neigh=check_neighs(npoints,nneigh,neigh)
  print ("Neighbours checked and eventually filtered")

if read_neighs:
  print ("Reading the neighbours")
  neighsarray = np.loadtxt(neighs_input_file)
  ncol=np.ma.size(neighsarray,axis=1)
  #if do_cutoff or do_kinetic_cont:
  maxneigh=ncol-2
  #if (ncol<maxneigh+2):
  #  print ("Error: number of columns in neighbour file is not consistent with the maximum number of neighbours")
  #  sys.exit()
  if (len(neighsarray)!=npoints):
    print ("Error: number of bins in neighbour file is not consistent")
    print (len(neighsarray),npoints)
    sys.exit()
  nneigh=neighsarray[:,1].astype(int)
  neigh=neighsarray[:,2:maxneigh+2].astype(int)
  #check neighs
  nneigh,neigh=check_neighs(npoints,nneigh,neigh)
  print ("Neighbours checked and eventually corrected")

# remove bad neighbours
if use_forces and wethreshold<0:

  neigh2=neigh
  nneigh2=nneigh
  nneigh=np.zeros((npoints),dtype=np.int32)
  neigh=np.ones((npoints,maxneigh),dtype=np.int32)
  neigh=-neigh
  for i in range(0,npoints):
     if nneigh2[i]>0:
       diff=tmparray[i,0:ndim]-tmparray[neigh2[i,0:nneigh2[i]],0:ndim]
       if totperiodic>0:
         diff=diff/box
         diff=diff-np.rint(diff)*period
         diff=diff*box
       dist2=(diff/width)*(diff/width)
       weighttot=weights[neigh2[i,0:nneigh2[i]],0:ndim]+weights[i,0:ndim]
       pippo=np.where(dist2>0,1,0)
       pippo2=np.where(weighttot==0,1,0)
       pippo3=np.where(pippo*pippo2==1,0,1)
       valid=np.amin(pippo3,axis=1)
       goodtrans=np.where(valid>0)
       nneigh[i]=len(goodtrans[0])
       if nneigh[i]>0:
         neigh[i,0:nneigh[i]]=neigh2[i,goodtrans[0]]

print ("Got the neighbours")

if do_cutoff:
  maxneigh=np.amax(nneigh)
prob=np.zeros((npoints,maxneigh))
fesdiff=np.empty((npoints,maxneigh))
fesdiff[:,:]=np.nan
logprobpath=np.empty((npoints,maxneigh))
logprobpath[:,:]=np.nan

if print_neighs:
  with open(neighs_file, 'a') as f:
      for i in range(0,npoints):
         f.write("%s %s " % (i,nneigh[i]))
         for j in range (0,maxneigh-1):
            f.write("%s " % (neigh[i,j]))
         f.write("%s \n" % (neigh[i,maxneigh-1]))

for i in range(0,npoints):
    diff=tmparray[i,0:ndim]-tmparray[neigh[i,0:nneigh[i]],0:ndim]
    if totperiodic>0:
      diff=diff/box
      diff=diff-np.rint(diff)*period
      diff=diff*box
    dist2=(diff/width)*(diff/width)
    dist=np.sum(dist2,axis=1)
    invdist=1/(np.power(dist,dexp))
    if use_forces:
      avergrad=tmparray[neigh[i,0:nneigh[i]],ndim:2*ndim]*weights[neigh[i,0:nneigh[i]],0:ndim]+tmparray[i,ndim:2*ndim]*weights[i,0:ndim]
      weighttot=weights[neigh[i,0:nneigh[i]],0:ndim]+weights[i,0:ndim]
      avergrad=np.where(weighttot>0,avergrad/weighttot,0)
      #pippo=np.where(diff>0,1,0)
      #pippo2=np.where(weighttot==0,1,0)
      #pippo3=np.where(pippo*pippo2==1,0,1)
      #valid=np.amin(pippo3,axis=1)
      energydiff=np.sum(avergrad*diff,axis=1)
    else:       
      energydiff=tmparray[i,ndim]-tmparray[neigh[i,0:nneigh[i]],ndim]
      #valid=np.where(np.isnan(energydiff),0,1)
    if do_mfepath or do_spath:
      logprobpath[i,0:nneigh[i]]=(np.log(invdist))+(energydiff/(2*kb*pathtemp)) 
      #logprobpath[i,0:nneigh[i]]=np.where(valid>0,(np.log(invdist))+(energydiff/(2*kb*pathtemp)),np.nan) 
    prob[i,0:nneigh[i]]=invdist*np.exp(energydiff/(2*kb*temp))
    fesdiff[i,0:nneigh[i]]=energydiff
    #prob[i,0:nneigh[i]]=np.where(valid>0,invdist*np.exp(energydiff/(2*kb*temp)),0.0)

#print("--- %s seconds ---" % (time.time() - start_time))
#sys.exit()

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
       for jj in range (0,nneigh[int(patharray[j-1,ndim])]):
          if neigh[int(patharray[j-1,ndim]),jj]==int(patharray[j,ndim]):
            refjj=jj
       minpathdefenerlike.append(-logprobpath[int(patharray[j-1,ndim]),refjj]) 
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
    validfreq=np.where(freq>0,1,0)
    initstate=np.argmax(minweights*validfreq)

  else:
    goodstates=np.where(nneigh>0)
    goodfes=fesarray[goodstates[0],ndim]
    initstate=goodstates[0][np.argmin(goodfes)]
  if do_rfd:
    print ("Free energy is calculated using reverse finite difference over KMC trajectory") 
    free_energy=run_kmc_rfd(nsteps,npoints,initstate)
    minfes=np.nanmin(free_energy)
    for nn in range (0,npoints):
       with open(free_energy_file, 'a') as f:
           if np.isnan(free_energy[nn]):
             for j in range (0,ndim):
                f.write("%s " % (tmparray[nn,j]))
             f.write("%s \n" % (np.nan))
           else:
             free_energy[nn]=free_energy[nn]-minfes     
             for j in range (0,ndim):
                f.write("%s " % (tmparray[nn,j]))
             f.write("%s \n" % (free_energy[nn]))
  else: 
    print ("Free energy is calculated from bin populations across KMC trajectory")
    pop=run_kmc(nsteps,npoints,initstate)  
      
    maxpop=np.amax(pop)
#for nn in range(0,npoints):
#   if pop[nn]>0:
#     names=str(tmparray[nn,0:ndim])
#     print names[1:-1],-kb*temp*np.log(pop[nn]/maxpop)

    free_energy=np.zeros((npoints))
   
    for nn in range (0,npoints):
       with open(free_energy_file, 'a') as f:
           if pop[nn]>0:
             free_energy[nn]=-kb*temp*np.log(pop[nn]/maxpop)
             for j in range (0,ndim):
                f.write("%s " % (tmparray[nn,j]))
             f.write("%s \n" % (free_energy[nn]))
           else:
             free_energy[nn]=np.nan
             for j in range (0,ndim):
                f.write("%s " % (tmparray[nn,j]))
             f.write("%s \n" % (np.nan))
    print ("Bin maximum population is:",maxpop)
# check and write accuracy

  error_count=0
  tot_error_diff=0
  max_error_diff=0 
  for nn in range(0,npoints):
     diff=tmparray[nn,0:ndim]-tmparray[neigh[nn,0:nneigh[nn]],0:ndim]
     if totperiodic>0:
       diff=diff/box
       diff=diff-np.rint(diff)*period
       diff=diff*box
     dist2=(diff/width)*(diff/width)
     dist=np.sum(dist2,axis=1)
     if use_forces:
       avergrad=tmparray[neigh[nn,0:nneigh[nn]],ndim:2*ndim]*weights[neigh[nn,0:nneigh[nn]],0:ndim]+tmparray[nn,ndim:2*ndim]*weights[nn,0:ndim]
       weighttot=weights[neigh[nn,0:nneigh[nn]],0:ndim]+weights[nn,0:ndim]
       avergrad=np.where(weighttot>0,avergrad/weighttot,0)
       energydiff=np.sum(avergrad*diff,axis=1)
     else:
       energydiff=tmparray[nn,ndim]-tmparray[neigh[nn,0:nneigh[nn]],ndim]
     energydiff_calc=free_energy[nn]-free_energy[neigh[nn,0:nneigh[nn]]]
     #pop_neighs=np.sum(pop[neigh[nn,0:nneigh[nn]]])
     error_diff=np.nanmean(np.absolute(energydiff_calc-energydiff))
     #if pop[nn]>0 and pop_neighs>0:
     if np.isnan(error_diff)==False:
       error_count=error_count+1
       if error_count==1:
         max_error_diff=error_diff
       if error_count>1:
         if error_diff>max_error_diff:
           max_error_diff=error_diff     
       tot_error_diff=tot_error_diff+error_diff

  print ("Average error on free energy differences between neighbor bins is:",tot_error_diff/error_count)   
  print ("Maximum error on free energy differences between neighbor bins is:",max_error_diff)
     
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
        if mc_mfepath:
          minpathdef=minpath
          minpathdeflnlike=minpathlnlike
          minpathdefenerlike=minpathenerlike
        elif minpathlnlike<minpathdeflnlike:
          minpathdef=minpath 
          minpathdeflnlike=minpathlnlike
          minpathdefenerlike=minpathenerlike
        with open(per_iter_path_file, 'a') as f:
            f.write("MINPATH: %s %s %s %s \n" % (minpathdeflnlike,np.sum(minpathdefenerlike),len(minpathdef),minpathlnlike-minpathdeflnlike))   
  
  with open(path_file, 'w') as f:
      f.write("# colvars, pathstate, free energy, deltaF \n")
      for nn in range (0,len(minpathdef)):
         for j in range (0,ndim):
            f.write("%s " % (tmparray[minpathdef[nn],j]))
         if nn==0:
           ener_tot_tmp=0.0
           if read_fes:
             f.write("%i %s %s \n" % (minpathdef[nn],fesarray[minpathdef[nn],ndim]," 0.0 "))
           else:
             f.write("%i %s %s \n" % (minpathdef[nn]," 0.0 "," 0.0 "))
         else:
           #ener_diff_tmp=(2*kb*pathtemp)*(-minpathdefenerlike[nn]+np.log(freqpath[minpathdef[nn-1],0]))
           lograte=-minpathdefenerlike[nn]+np.log(freqpath[minpathdef[nn-1],0])
           for jj in range (0,nneigh[minpathdef[nn]]):    
              if neigh[minpathdef[nn],jj]==minpathdef[nn-1]:
                refjj=jj
           revlograte=logprobpath[minpathdef[nn],refjj]+np.log(freqpath[minpathdef[nn],0]) 
           ener_diff_tmp=(kb*pathtemp)*(lograte-revlograte)
           ener_tot_tmp=ener_tot_tmp-ener_diff_tmp
           if read_fes:
             f.write("%i %s %s \n" % (minpathdef[nn],fesarray[minpathdef[nn],ndim],ener_diff_tmp))
           else:
             f.write("%i %s %s \n" % (minpathdef[nn],ener_tot_tmp,ener_diff_tmp))

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
     this_max = -1
     init=True
     for i in range (0,ngrp):
        this = atnum[i]
        for j in range (0,nneigh[this]):
           if np.isnan(logprobpath[this,j]):
             continue  
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
       ener_like=e_lnprob[this]
       len_path=1
       with open(path_file, 'w') as f:
           f.write("# colvars, pathstate, free energy, deltaF \n")
           for j in range (0,ndim):
              f.write("%s " % (tmparray[this,j]))
           #f.write("%i %s %s \n" % (this,fesarray[this,ndim],e_lnprob[this]))
           ener_tot_tmp=0.0
           if read_fes: 
             f.write("%i %s %s \n" % (this,fesarray[this,ndim]," 0.0 "))
           else:
             f.write("%i %s %s \n" % (this," 0.0 "," 0.0 ")) 
           while this1!=startbinmfepath:
                len_path=len_path+1
                this=come_from[this1]
                for j in range (0,ndim):
                   f.write("%s " % (tmparray[this,j]))
                for jj in range (0,nneigh[this]): 
                   if neigh[this,jj]==this1: 
                     refjj=jj
                for jj in range (0,nneigh[this1]):
                   if neigh[this1,jj]==this:
                     revrefjj=jj
                lograte=logprobpath[this,refjj]+np.log(freqpath[this,0])
                revlograte=logprobpath[this1,revrefjj]+np.log(freqpath[this1,0])
                ener_diff_tmp=(kb*pathtemp)*(lograte-revlograte)
                ener_tot_tmp=ener_tot_tmp+ener_diff_tmp
                if read_fes:  
                  f.write("%i %s %s \n" % (this,fesarray[this,ndim],-ener_diff_tmp))
                else:
                  f.write("%i %s %s \n" % (this,ener_tot_tmp,-ener_diff_tmp))
                this1=this
       with open(per_iter_path_file, 'a') as f:
           f.write("%s %s \n" % (-ener_like,len_path))

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
