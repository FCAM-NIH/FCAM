import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
import time
start_time = time.time()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--inputfile", \
                        help="input file for analysis", \
                        type=str, required=True)
    parser.add_argument("-temp", "--temp", help="Temperature (in Kelvin) for calculating the force constant (k) according to half bin width (k=4*kb*temp/(width*width)) ", \
                        default=-1.0,type=float, required=False)
    parser.add_argument("-kb", "--kb", help="Boltzmann factor for calculating the force constant (k) and defining free energy units. Default is 0.00831... kJ/mol", \
                        default=0.00831446261815324,type=float, required=False)
    parser.add_argument("-colgener", "--colgener", help="Column to read the Gaussian energy in the COLVAR file for filtering (Default is the second last)", \
                        default=-1,type=int, required=False)
    parser.add_argument("-valgefilt", "--valgefilt", help="Difference threshold between calculated gaussian energy and the one reported in the COLVAR file for filtering ", \
                        default=1000.0,type=float, required=False)
    parser.add_argument("-nometaf","--nocalcmetabiasforce", \
                        help="Do not calculate bias forces of metadynamics Gaussian hills. By default metadynamics bias calculation is ON", \
                        default=True, dest='do_hbias', action='store_false')
    parser.add_argument("-obgf", "--outbiasgradfile", \
                        help="output file of bias gradients for each frame", \
                        default="bias_grad.out",type=str, required=False)
    parser.add_argument("-oepf", "--outeffpointsfile", \
                        help="output file of effective points to calculate forces", \
                        default="eff_points.out",type=str, required=False)    
    parser.add_argument("-oeff", "--outeffforcefile", \
                        help="output file effective points and forces", \
                        default="force_on_eff_points.out",type=str, required=False)
    parser.add_argument("-ocmbeff", "--outcombeffforcefile", \
                        help="output combined file effective points and forces", \
                        default="force_on_eff_points_comb.out",type=str, required=False)
    parser.add_argument("-obff", "--outbinforcefile", \
                        help="output file of binned colvar and forces", \
                        default="force_on_bin_points.out",type=str, required=False)
    parser.add_argument("-nobdat","--nobindata", \
                        help="Do not bin data according to provided grid", \
                        default=True, dest='do_bdat', action='store_false')
    parser.add_argument("-nobound","--noboundaries", \
                        help="Do not exclude data beyond grid boundaries", \
                        default=True, dest='do_bound', action='store_false')
    parser.add_argument("-gefilt","--gaussianenergyfilter", \
                        help="Filter data by comparing the calculated gaussian energy with the one reported in the COLVAR file according to -valgefilt", \
                        default=False, dest='do_gefilt', action='store_true')
    parser.add_argument("-nofeffpc","--nofasteffpointcalc", \
                        help="Do not use algorithm for fast calculation of effective points (through binning)", \
                        default=True, dest='do_feffpc', action='store_false')
    parser.add_argument("-noeffpb","--noeffpointbin", \
                        help="Do not calculate effective points by binning but use non-overlapping spherical domains", \
                        default=True, dest='do_effpb', action='store_false')
    parser.add_argument("-nofbind","--nofastbindata", \
                        help="Do not use algorithm for fast data binning", \
                        default=True, dest='do_fbind', action='store_false')
    parser.add_argument("-jceffp","--justcalceffpoints", \
                        help="Calculate effective points and do nothing else. COLVARS and GRID data must at least be provided", \
                        default=False, dest='do_jceffp', action='store_true')
    parser.add_argument("-jcmetab","--justcalcmetabias", \
                        help="Calculate metadynamics bias potential and do nothing else. COLVARS, HILLS and GRID data must at least be provided", \
                        default=False, dest='do_jmetab', action='store_true')
    parser.add_argument("-hlfl","--hillfreqlarge", \
                        help="Metadynamics in which HILLS are stored more frequently than COLVARS", \
                        default=False, dest='do_hlfl', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()

# parameters
hcutoff=6.25 # cutoff for Gaussians
wcutoff=6.25 # cutoff for Gaussians in weight calculation

# Variables
ifile=args.inputfile
do_hills_bias=args.do_hbias
do_bin_data=args.do_bdat
do_boundaries=args.do_bound
do_gefilter=args.do_gefilt
do_just_eff_points=args.do_jceffp
do_just_hills_bias=args.do_jmetab
do_fast_eff_p_calc=args.do_feffpc
do_bin_eff_p_calc=args.do_effpb
do_fast_bin_data=args.do_fbind
bias_grad_file=args.outbiasgradfile
eff_points_file=args.outeffpointsfile
force_points_file=args.outeffforcefile
force_points_file_comb=args.outcombeffforcefile
force_bin_file=args.outbinforcefile
temp=args.temp
kb=args.kb
tgefilt=args.valgefilt
colgener=args.colgener
do_large_hfreq=args.do_hlfl

calc_epoints=True # True unless is read from input
calc_force_eff=True # True unless is read from input

ncolvars=0
ndim=0
ngfiles=0
nefiles=0
nffiles=0
read_gfile=False
read_efile=False
read_ffile=False
has_hills=False
nactive=0
cfile='none'
hfile='none'

# INPUT PARSER

f=open (ifile, 'r') 
for line in f:
    parts=line.split()
    nparts=len(parts)
    has_c=False
    ncfiles=0
    nhfiles=0
    nafields=0 
    for i in range (0,nparts):
       if str(parts[i])=="COLVAR_FILE": 
         lc=i
         has_c=True
         ncfiles=ncfiles+1
    if ncfiles>1:
      print ("ERROR, COLVAR_FILE must be specidied just one time on a single line") 
      sys.exit()
    if has_c:
      if ncolvars==0:
        cfile=[str(parts[lc+1])]
        has_h=False 
        for i in range (0,nparts):
           if str(parts[i])=="HILLS_FILE":
             lh=i
             has_h=True 
             nhfiles=nhfiles+1
        if nhfiles>1:
          print ("ERROR, HILLS_FILE must be specidied just one time on a single line")
          sys.exit()
        if has_h:
          has_hills=[True]
          hfile=[str(parts[lh+1])]  
          has_a=False
          nact=0 
          for i in range (0,nparts):
             if str(parts[i])=="HILLS_CVS": 
               la=i
               has_a=True
               nafields=nafields+1
          if has_a==False or nafields>1:
            print ("ERROR, HILLS_CVS not specified or specidied more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break  
             nact=nact+1
          nactive=[nact]
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0
          for i in range(la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1 
          a_cvs=[pippo]
        else:
          has_hills=[False]
          hfile=['none']
          nactive=[int(0)] 
          a_cvs=[int(-1)]
      if ncolvars>0:
        cfile.append(str(parts[lc+1]))
        has_h=False
        for i in range (0,nparts):
           if str(parts[i])=="HILLS_FILE":
             lh=i
             has_h=True
             nhfiles=nhfiles+1
        if nhfiles>1:
          print ("ERROR, HILLS_FILE must be specidied just one time on a single line")
          sys.exit()
        if has_h:
          has_hills.append(True)
          hfile.append(str(parts[lh+1]))
          has_a=False
          nact=0   
          for i in range (0,nparts):
             if str(parts[i])=="HILLS_CVS": 
               la=i
               has_a=True
               nafields=nafields+1
          if has_a==False or nafields>1:
            print ("ERROR, HILLS_CVS not specified or specidied more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break 
             nact=nact+1
          nactive.append(nact)
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0
          for i in range(la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1
          a_cvs.append(pippo)
        else:
          has_hills.append(False)
          hfile.append('none')
          nactive.append(int(0))
          a_cvs.append(int(-1))
      ncolvars=ncolvars+1
    if str(parts[0])=="CV":
      if ndim==0:
        whichcv=[int(parts[1])-1]
        lowbound=[float(parts[2])]
        upbound=[float(parts[3])]
        if float(parts[2])>=float(parts[3]):
          print ("ERROR: lower boundary must be smaller than upper boudary ") 
          sys.exit() 
        npointsv=[int(parts[4])]
        if len(parts)>5:
          if str(parts[5])=="PERIODIC":
            periodic=[1] 
          else:
            print ("Unrecognized character:",str(parts[5]))
            sys.exit()         
        else:
          periodic=[0]
      if ndim>0:
        whichcv.append(int(parts[1])-1)
        lowbound.append(float(parts[2]))
        upbound.append(float(parts[3]))
        if float(parts[2])>=float(parts[3]):
          print ("ERROR: lower boundary must be smaller than upper boudary ")
          sys.exit()
        npointsv.append(int(parts[4]))
        if len(parts)>5: 
          if str(parts[5])=="PERIODIC":
            periodic.append(1)
          else:
            print ("Unrecognized character:",str(parts[5]))
            sys.exit()
        else:
          periodic.append(0)
      ndim=ndim+1       
    if str(parts[0])=="READ_GRAD":
      if ngfiles==0:
        gfile=[str(parts[1])] 
        read_gfile=True
        do_hills_bias=False
      if ngfiles>0:
        gfile.append(str(parts[1]))      
      ngfiles=ngfiles+1
    if str(parts[0])=="READ_EPOINTS":
      if nefiles==0:
        efile=[str(parts[1])]
        read_efile=True
        calc_epoints=False
      if nefiles>0:
        efile.append(str(parts[1])) 
      nefiles=nefiles+1  
    if str(parts[0])=="READ_FORCE":
      if nffiles==0:
        ffile=[str(parts[1])]
        read_ffile=True        
        read_efile=False
        do_hills_bias=False 
        calc_force_eff=False
        calc_epoints=False 
      if nffiles>0:
        ffile.append(str(parts[1]))
      nffiles=nffiles+1

print ("Input read")
if do_just_eff_points:
  calc_epoints=True
  do_hills_bias=False
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False  
  do_bin_data=False

if do_just_hills_bias:
  calc_epoints=False
  do_hills_bias=True
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False
  do_bin_data=False
  
if ncolvars==0:
  calc_epoints=False  
  calc_force_eff=False

if ndim==0:
  print ("ERROR: number of variables is zero, please provide some to continue")
  sys.exit()
       
if calc_force_eff and temp<0:
  print ("ERROR: temperature for calculating forces not provided or negative value")
  sys.exit()
 
if read_gfile:
  if ngfiles!=1:
    if ngfiles!=ncolvars:
      print ("ERROR: please provide a unique gradient file")
      print ("or a gradient file for each colvar.")
      print ("Note that in either case this must be consistent with the ORDERED set of colvar files provided.")
      sys.exit()

if do_hills_bias:
  with open(bias_grad_file, 'w') as f:
      f.write("# Time, grad,Gaussenergy,rep \n")

if calc_epoints:
  with open(eff_points_file, 'w') as f:
      f.write("# numeff, colvar, rep \n")

has_hills=np.array(has_hills)
nactive=np.array(nactive)
upbound=np.array(upbound)
lowbound=np.array(lowbound)
npointsv=np.array(npointsv)
periodic=np.array(periodic,dtype=np.int8)
box=upbound-lowbound
width=box/npointsv

if calc_force_eff:
  with open(force_points_file, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]/2.0))
         f.write("  %s  " % (2*npointsv[j]))
         f.write("  %s  \n" % periodic[j])

if read_ffile:
  with open(force_points_file_comb, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]/2.0))
         f.write("  %s  " % (2*npointsv[j]))
         f.write("  %s  \n" % periodic[j])

if do_bin_data:
  with open(force_bin_file, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % width[j])
         f.write("  %s  " % npointsv[j])
         f.write("  %s  \n" % periodic[j])

cunique=np.unique(cfile)
hunique=np.unique(hfile)
vunique=np.unique(whichcv)

if len(cunique)!=len(cfile) and cunique[0]!="none":
  print ("ERROR: same COLVAR file introduced multiple times in the input")
  sys.exit()

if len(hunique)!=len(hfile) and hunique[0]!="none":
  print ("ERROR: same HILLS file introduced multiple times in the input")
  sys.exit()

if len(vunique)!=len(whichcv):
  print ("ERROR: same CV introduced multiple times in the input")
  sys.exit()

iactive=np.zeros((ncolvars,ndim),dtype=np.int8)
iactive[:,:]=-1

allfound=True
for i in range (0,ncolvars):
   if nactive[i]>ndim:
     print ("ERROR: number of HILLS_CVS larger than total number of CVS")
     sys.exit() 
   for j in range (0,nactive[i]):
      if nactive[i]>0: 
        if a_cvs[i][j]>=0: 
          hasit=False 
          for k in range (0,ndim):
             if a_cvs[i][j]==whichcv[k]:
               hasit=True
               iactive[i,j]=k
          if hasit==False:
            allfound=False 

if allfound==False:
  print ("ERROR: HILLS_CVS must be part of the CVS used for CV")
  sys.exit()

# read the hills files
for i in range (0,ncolvars):
   if has_hills[i]:
     if i==0:
       hillsarray=[np.loadtxt(hfile[i])]
       nhills=[len(hillsarray[i])]
   
     else:

       hillsarray.append(np.loadtxt(hfile[i]))
       nhills.append(len(hillsarray[i]))

   else:     
     if i==0:
       hillsarray=[0]
       nhills=[0]
       #tmp_cvhills=[0]
       #tmp_deltahills=[0]
       #whills=[0]
     else:
       hillsarray.append(0)
       nhills.append(0)

# read the colvar file
for i in range (0,ncolvars):
   if i==0:
     colvarsarray=[np.loadtxt(cfile[i])]
     npoints=[len(colvarsarray[i])]
     gradv=[np.zeros((npoints[i],ndim))]
   else:
     colvarsarray.append(np.loadtxt(cfile[i]))
     npoints.append(len(colvarsarray[i]))
     gradv.append(np.zeros((npoints[i],ndim)))

if do_gefilter:
  for i in range (0,ncolvars):
     if i==0:
       if colgener<0:
         colgenerread=[np.ma.size(colvarsarray[i],axis=1)-2]
       else:
         colgenerread=[colgener-1]  
     else:
       if colgener<0:
         colgenerread.append(np.ma.size(colvarsarray[i],axis=1)-2)
       else:
         colgenerread.append(colgener-1)

# ROUTINE TO CALCULATE THE NUMBER OF EFFECTIVE POINTS (USEFUL TO REDUCE NUMBER OF POINTS AND THUS FORCE CALCULATIONS)

def calc_eff_points(numpoints, inputarray, npointsins):
   diffc=np.zeros((numpoints,ndim))
   distance=np.zeros((numpoints))
   numinbin=np.zeros((numpoints),dtype=np.int64)
   neffp=1 
   ceff=np.zeros((numpoints,ndim))
   ceff[0,:]=inputarray[0,:]
   numinbin[0]=1
   totperiodic=np.sum(periodic[0:ndim])
   for i in range(1,numpoints):
      diffc[0:neffp,:]=ceff[0:neffp,:]-inputarray[i,:]
      if totperiodic>0:
        diffc[0:neffp,:]=diffc[0:neffp,:]/box[0:ndim]
        diffc[0:neffp,:]=diffc[0:neffp,:]-np.rint(diffc[0:neffp,:])*periodic[0:ndim]
        diffc[0:neffp,:]=diffc[0:neffp,:]*box[0:ndim]
      diffc[0:neffp,:]=2.0*diffc[0:neffp,:]/width[0:ndim]
      diffc[0:neffp,:]=diffc[0:neffp,:]*diffc[0:neffp,:]
      distance[0:neffp]=np.sum(diffc[0:neffp,:],axis=1)
      whichbin=np.argmin(distance[0:neffp])
      mindistance=distance[whichbin]
      if mindistance>1:
        ceff[neffp,:]=inputarray[i,:]
        whichbin=neffp
        neffp=neffp+1
      numinbin[whichbin]=numinbin[whichbin]+npointsins[i]
   return ceff[0:neffp], neffp, numinbin[0:neffp]

# try to do a faster routine 

def calc_eff_points_bins(numepoints, effparray, npointsins):
   colvarsbineff=np.zeros((numepoints, ndim))
   nbins=1
   mywidth=width/2.0
   diffc=np.zeros((numepoints,ndim))
   distance=np.zeros((numepoints))
   diffbin=np.zeros(ndim)
   colvarbin=np.zeros(ndim)
   numinbin=np.zeros((numepoints),dtype=np.int64)
   totperiodic=np.sum(periodic[0:ndim])
   colvarsbineff[0,:]=lowbound+0.5*box+0.5*mywidth
   for i in range(0,numepoints):
      diffc[i,:]=effparray[i,:]-(lowbound+0.5*box)
      if totperiodic>0:
        diffc[i,:]=diffc[i,:]/box[0:ndim]
        diffc[i,:]=diffc[i,:]-np.rint(diffc[i,:])*periodic[0:ndim]
        diffc[i,:]=diffc[i,:]*box[0:ndim]
      colvarbin=0.5*mywidth+mywidth*np.floor(diffc[i,:]/mywidth)+lowbound+0.5*box
      colvarbin=(colvarbin-(lowbound+0.5*box))/box[0:ndim]
      colvarbin=colvarbin-np.rint(colvarbin)*periodic[0:ndim]
      colvarbin=colvarbin*box[0:ndim]+(lowbound+0.5*box)
      diffc[0:nbins,:]=colvarsbineff[0:nbins,:]-colvarbin[:]
      if totperiodic>0:
        diffc[0:nbins,:]=diffc[0:nbins,:]/box[0:ndim]
        diffc[0:nbins,:]=diffc[0:nbins,:]-np.rint(diffc[0:nbins,:])*periodic[0:ndim]
        diffc[0:nbins,:]=diffc[0:nbins,:]*box[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]/mywidth[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]*diffc[0:nbins,:]
      distance[0:nbins]=np.sum(diffc[0:nbins,:],axis=1)
      whichbin=np.argmin(distance[0:nbins])
      mindistance=distance[whichbin]
      if mindistance>0.5:
        colvarsbineff[nbins,:]=colvarbin
        whichbin=nbins
        nbins=nbins+1
      numinbin[whichbin]=numinbin[whichbin]+npointsins[i]
   return colvarsbineff, nbins, numinbin[0:nbins]

# fast effective points using fast data binning (not always reliable but very fast, especially for high dimensionality)

def fast_calc_eff_points(numepoints, effparray, npointsins):
   diffc=np.zeros((numepoints,ndim))
   myshift=np.ones((ndim),dtype=np.int64)
   mywidth=width/2.0
   shift=1
   myshift[0]=1
   binid=np.ones((numepoints),dtype=np.int64)
   for i in range (1,ndim):
      #myshift[i]=2*shift*(2*npointsv[i])
      myshift[i]=shift*2*(npointsv[i])
      shift=myshift[i]
   totperiodic=np.sum(periodic[0:ndim])
   diffc[:,:]=effparray[:,:]-(lowbound+0.5*box) 
   if totperiodic>0:
     diffc[:,:]=diffc[:,:]/box[0:ndim]
     diffc[:,:]=diffc[:,:]-np.rint(diffc[:,:])*periodic[0:ndim]
     diffc[:,:]=diffc[:,:]*box[0:ndim]
   bingrid=np.floor(diffc[:,:]/mywidth) 
   binid=np.sum(bingrid*myshift,axis=1)
   effbinid=np.unique(binid,return_index=True,return_inverse=True,return_counts=True)
   neffp=effbinid[1].size
   indexbin=np.argsort(effbinid[2][:])
   effcount=np.cumsum(effbinid[3])
   effindexbin=np.split(indexbin,effcount)
   numinpoints=np.zeros((neffp),dtype=np.int64)
   for i in range (0,neffp):
      numinpoints[i]=np.sum(npointsins[effindexbin[i]])
   colvarbin=0.5*mywidth+mywidth*bingrid[effbinid[1][:],:]+(lowbound+0.5*box)
   if totperiodic>0:
     colvarbin=(colvarbin-(lowbound+0.5*box))/box[0:ndim]
     colvarbin=colvarbin-np.rint(colvarbin)*periodic[0:ndim]
     colvarbin=colvarbin*box[0:ndim]+(lowbound+0.5*box)     
   return colvarbin[0:neffp,0:ndim], neffp, numinpoints[0:neffp]  

#ROUTINE TO CALCULATE THE FORCE ON A SET OF POINTS

def calc_vhar_force(numepoints, numpoints, effparray, colvars, gradbias):
   grade=np.zeros((numepoints,ndim))
   vargrade=np.zeros((numepoints,ndim))
   diffc=np.zeros((numpoints,ndim))
   tweights=np.zeros((numepoints))
   totperiodic=np.sum(periodic[0:ndim]) 
   for i in range(0,numepoints):
      diffc=colvars[:,:]-effparray[i,:]
      if totperiodic>0:
        diffc=diffc/box[0:ndim]        
        diffc=diffc-np.rint(diffc)*periodic[0:ndim]
        diffc=diffc*box[0:ndim]
      diffc=2.0*diffc/width[0:ndim]
      distance=0.5*np.sum(diffc[0:numpoints,:]*diffc[0:numpoints,:],axis=1)
      whichpoints=np.where(distance<wcutoff) 
      #whichpoints=np.array(whichpoints)
      #whichpoints=np.ndarray.flatten(whichpoints) 
      weight=np.exp(-distance[whichpoints[0][:]])
      #weight=np.exp(np.sum(-0.5*diffc[0:numpoints,:]*diffc[0:numpoints,:],axis=1))
      if weight.size>0:
        #if np.amax(weight)>0:  
      #for j in range(0,ndim):
        #grade[i,j]=-np.average(2.0*kb*temp*diffc[whichpoints[:],j]/width[j]+gradbias[whichpoints[:],j],weights=weight[:]) 
      #grade[i,:]=-np.sum((2.0*kb*temp*diffc[0:numpoints,:]/width[:]+gradbias[0:numpoints,:])*weight[0:numpoints,np.newaxis],axis=0)
        instgrad=(2.0*kb*temp*diffc[whichpoints[0][:],:]/width[:]+gradbias[whichpoints[0][:],:])
        grade[i,:]=-np.sum(instgrad[:,:]*weight[:,np.newaxis],axis=0) 
        vargrade[i,:]=np.sum(instgrad[:,:]*instgrad[:,:]*weight[:,np.newaxis],axis=0)
        tweights[i]=np.sum(weight)
        if tweights[i]>0.0:
          grade[i,:]=grade[i,:]/tweights[i] 
          vargrade[i,:]=vargrade[i,:]/tweights[i]
          vargrade[i,:]=(vargrade[i,:]-grade[i,:]*grade[i,:])/tweights[i]
   for j in range(0,ndim):       
      vargrade[:,j]=np.where(tweights[:]>3.0,vargrade[:,j],0.0) # The threshold 3 can be put in input
   #   vargrade[:,j]=np.where(vargrade[:,j]>0,1/vargrade[:,j],0.0)
   return grade,np.where(vargrade>0,1/vargrade,0.0)

# ROUTINE TO BIN THE DATA STARTING FROM FIRST POINT

def bin_data(numepoints, effparray, weights, gradarray):
   colvarsbineff=np.zeros((numepoints, ndim))
   nbins=1
   diffc=np.zeros((numepoints,ndim)) 
   distance=np.zeros((numepoints))
   gradbin=np.zeros((numepoints,ndim))
   diffbin=np.zeros(ndim)
   colvarbin=np.zeros(ndim)
   weightbin=np.zeros((numepoints,ndim))
   numinbin=np.zeros((numepoints),dtype=np.int64)
   #indexmax=np.argmax(weights)
   indexmax=0
   totperiodic=np.sum(periodic[0:ndim])
   #colvarsbineff[0,:]=effparray[indexmax,:]
   colvarsbineff[0,:]=lowbound+0.5*box+0.5*width
   for i in range(0,numepoints):
      #diffc[i,:]=effparray[i,:]-(effparray[indexmax,:]-0.5*width)
      diffc[i,:]=effparray[i,:]-(lowbound+0.5*box)
      if totperiodic>0:
        diffc[i,:]=diffc[i,:]/box[0:ndim]
        diffc[i,:]=diffc[i,:]-np.rint(diffc[i,:])*periodic[0:ndim]
        diffc[i,:]=diffc[i,:]*box[0:ndim]
      colvarbin=0.5*width+width*np.floor(diffc[i,:]/width)+lowbound+0.5*box
      colvarbin=(colvarbin-(lowbound+0.5*box))/box[0:ndim]
      colvarbin=colvarbin-np.rint(colvarbin)*periodic[0:ndim]
      colvarbin=colvarbin*box[0:ndim]+(lowbound+0.5*box)
      diffc[0:nbins,:]=colvarsbineff[0:nbins,:]-colvarbin[:]
      if totperiodic>0:
        diffc[0:nbins,:]=diffc[0:nbins,:]/box[0:ndim]
        diffc[0:nbins,:]=diffc[0:nbins,:]-np.rint(diffc[0:nbins,:])*periodic[0:ndim]
        diffc[0:nbins,:]=diffc[0:nbins,:]*box[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]/width[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]*diffc[0:nbins,:]
      distance[0:nbins]=np.sum(diffc[0:nbins,:],axis=1)
      whichbin=np.argmin(distance[0:nbins])
      mindistance=distance[whichbin]  
      if mindistance>0.5:
        colvarsbineff[nbins,:]=colvarbin
        whichbin=nbins
        nbins=nbins+1
      numinbin[whichbin]=numinbin[whichbin]+1
      gradbin[whichbin,:]=gradbin[whichbin,:]+gradarray[i,:]*weights[i,:]
      #gradbin[whichbin,:]=gradbin[whichbin,:]+gradarray[i,:]
      weightbin[whichbin,:]=weightbin[whichbin,:]+weights[i,:]   
   for j in range(0,ndim):
      gradbin[0:nbins,j]=np.where(weightbin[0:nbins,j]>0,gradbin[0:nbins,j]/weightbin[0:nbins,j],0)
      #gradbin[0:nbins,j]=np.where(weightbin[0:nbins]>0,gradbin[0:nbins,j]/numinbin[0:nbins],0) 
   #weightbin[0:nbins]=np.where(numinbin[0:nbins]>0,weightbin[0:nbins]/numinbin[0:nbins],0)
   return colvarsbineff, gradbin, weightbin, nbins

# Fast bin of the data based on bin identifier

def fast_bin_data(numepoints, effparray, weights, gradarray):
   diffc=np.zeros((numepoints,ndim))
   myshift=np.ones((ndim),dtype=np.int64)
   binid=np.ones((numepoints),dtype=np.int64) 
   mywidth=width
   shift=1
   myshift[0]=1
   for i in range (1,ndim):
      myshift[i]=shift*(npointsv[i]+2)
      shift=myshift[i]
   totperiodic=np.sum(periodic[0:ndim])
   diffc[:,:]=effparray[:,:]-(lowbound+0.5*box)
   if totperiodic>0:
     diffc[:,:]=diffc[:,:]/box[0:ndim]
     diffc[:,:]=diffc[:,:]-np.rint(diffc[:,:])*periodic[0:ndim]
     diffc[:,:]=diffc[:,:]*box[0:ndim]
   bingrid=np.floor(diffc[:,:]/mywidth)
   binid=np.sum((bingrid[:,0:ndim])*myshift[np.newaxis,0:ndim],axis=1)
   effbinid=np.unique(binid,return_index=True,return_inverse=True,return_counts=True)
   neffp=effbinid[1].size
   gradbin=np.zeros((neffp,ndim))
   weightbin=np.zeros((neffp,ndim))
   indexbin=np.argsort(effbinid[2][:])
   effcount=np.cumsum(effbinid[3])
   effindexbin=np.split(indexbin,effcount)
   for i in range (0,neffp):
      gradbin[i,:]=np.sum(gradarray[effindexbin[i],:]*weights[effindexbin[i],:],axis=0)
      weightbin[i,:]=np.sum(weights[effindexbin[i],:],axis=0) 
   for j in range (0,ndim):
      gradbin[:,j]=np.where(weightbin[:,j]>0,gradbin[:,j]/weightbin[:,j],0)
   colvarbin=0.5*mywidth+mywidth*bingrid[effbinid[1][:],:]+(lowbound+0.5*box)
   if totperiodic>0:
     colvarbin=(colvarbin-(lowbound+0.5*box))/box[0:ndim]
     colvarbin=colvarbin-np.rint(colvarbin)*periodic[0:ndim]
     colvarbin=colvarbin*box[0:ndim]+(lowbound+0.5*box)
   return colvarbin[0:neffp,0:ndim], gradbin, weightbin, neffp

# calc HILLS forces

if do_hills_bias:
  print ("Calculating metadynamics bias forces on each COLVAR point of the selected variables from the HILLS files...")
  for k in range (0,ncolvars):
     if k==0:
       gaussenergy=[np.zeros((npoints[k]))]
     else:
       gaussenergy.append(np.zeros((npoints[k])))
     diff=np.zeros((nhills[k],nactive[k]))
     diff2=np.zeros((nhills[k],nactive[k]))
     expdiff=np.zeros((nhills[k]))
     whichhills=np.zeros((nhills[k]),dtype=np.int64)
     trh=0
     index=0
     dvec=np.arange(nhills[k])
     if nactive[k]>0:
       countinter=0
       numhills=0 
       for i in range(0,npoints[k]):
          whichhills_old=whichhills
          if i>0 and colvarsarray[k][i,0]<colvarsarray[k][i-1,0]:
            if hillsarray[k][numhills-1,0]>hillsarray[k][numhills,0]: 
              index=trh
              whichhills_old=np.where(dvec<trh,1,0)
          if do_large_hfreq:
            if numhills>=trh:
              index=trh
              whichhills_old=np.where(dvec<trh,1,0)
          trh=1+index+np.array(np.where(hillsarray[k][index+1:nhills[k],0]-hillsarray[k][index:nhills[k]-1,0]<0))
          if trh.size!=0:
            trh=np.amin(2+index+np.array(np.where(hillsarray[k][index+2:nhills[k],0]-hillsarray[k][index+1:nhills[k]-1,0]<0)))
          else:
            trh=nhills[k]
          whichhills=np.where(hillsarray[k][:,0]<=colvarsarray[k][i,0],1,0)
          whichhills=np.where(dvec>=trh,0,whichhills)
          whichhills=np.where(whichhills_old>0,1,whichhills)
          numhills=np.sum(whichhills)
          if numhills>0:
            diff[0:numhills,0:nactive[k]]=hillsarray[k][0:numhills,1:nactive[k]+1]-colvarsarray[k][i,a_cvs[k][0:nactive[k]]+1]
            diff[0:numhills,0:nactive[k]]=diff[0:numhills,0:nactive[k]]/box[iactive[k,0:nactive[k]]]
            diff[0:numhills,0:nactive[k]]=diff[0:numhills,0:nactive[k]]-np.rint(diff[0:numhills,0:nactive[k]])*periodic[iactive[k,0:nactive[k]]]
            diff[0:numhills,0:nactive[k]]=diff[0:numhills,0:nactive[k]]*box[iactive[k,0:nactive[k]]]
            diff[0:numhills,0:nactive[k]]=diff[0:numhills,0:nactive[k]]/hillsarray[k][0:numhills,nactive[k]+1:2*nactive[k]+1]
            diff2[0:numhills,0:nactive[k]]=diff[0:numhills,0:nactive[k]]*diff[0:numhills,0:nactive[k]]
            diff2[0:numhills,0:nactive[k]]=0.5*diff2[0:numhills,0:nactive[k]]
            expdiff[0:numhills]=np.sum(diff2[0:numhills,0:nactive[k]],axis=1)
            expdiff[0:numhills]=np.where(expdiff[0:numhills]<hcutoff,np.exp(-expdiff[0:numhills]),0.0)
            expdiff[0:numhills]=hillsarray[k][0:numhills,2*nactive[k]+1]*expdiff[0:numhills]
            gaussenergy[k][i]=np.sum(expdiff[0:numhills])
            for j in range(0,nactive[k]):  
               gradv[k][i,iactive[k,j]]=np.sum(diff[0:numhills,j]*expdiff[0:numhills]/hillsarray[k][0:numhills,nactive[k]+1+j],axis=0)
          with open(bias_grad_file, 'a') as f:
              f.write("%s " % (colvarsarray[k][i,0]))
              for j in range(0,ndim):
                 f.write("%s " % (gradv[k][i,j]))
              f.write("%s " % gaussenergy[k][i])
              f.write("%s " % numhills)
              f.write("%s " % trh)
              f.write("%s " % index)
              f.write("%s \n" % (k))
     else:
       for i in range(0,npoints[k]):
          with open(bias_grad_file, 'a') as f:
              f.write("%s " % (colvarsarray[k][i,0]))
              for j in range(0,ndim):
                 f.write("%s " % (gradv[k][i,j]))
              f.write("%s " % " 0 ")
              f.write("%s " % " 0 " )
              f.write("%s " % " 0 ")
              f.write("%s " % " 0 ")
              f.write("%s \n" % (k))

# READ EXTERNAL FILE WITH GRADIENTS
       
if read_gfile:
  print ("Reading bias forces from external file...") 
  if ngfiles==1: 
    gradarray = [np.loadtxt(gfile[0])]
    if np.sum(npoints[:])!=len(gradarray[0]):
      print ("ERROR: gradient file doesn't match COLVAR files")
      sys.exit() 
    totpoints=0
    for k in range (0,ncolvars):
       if k==0:
         gradv=[gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1]]
         gaussenergy=[gradarray[0][totpoints:totpoints+npoints[k],ndim+1]]
       if k>0:
         gradv.append(gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1])
         gaussenergy.append(gradarray[0][totpoints:totpoints+npoints[k],ndim+1]) 
       totpoints=totpoints+npoints[k]
  else:
    for k in range (0,ngfiles):
       if npoints[k]!=len(gradarray[k]):
         print ("ERROR: gradient file doesn't match COLVAR file")
         sys.exit() 
       if k==0:    
         gradv=[gradarray[:,1:ndim+1]]
         gaussenergy=[gradarray[:,ndim+1]]
       if k>0:
         gradv.append(gradarray[:,1:ndim+1]) 
         gaussenergy.append(gradarray[:,ndim+1])

# create masked colvarsarray and gradv eliminating values beyond boundaries
if do_boundaries:
  for i in range (0,ncolvars):
     tmpcolvarsarray=colvarsarray[i]
     for j in range(0,ndim):
        gradv[i][0:npoints[i],j]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]<lowbound[j],'NaN',gradv[i][0:npoints[i],j])
        gradv[i][0:npoints[i],j]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]>upbound[j],'NaN',gradv[i][0:npoints[i],j])
        colvarsarray[i][0:npoints[i],whichcv[j]+1]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]<lowbound[j],'NaN',tmpcolvarsarray[0:npoints[i],whichcv[j]+1])
        colvarsarray[i][0:npoints[i],whichcv[j]+1]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]>upbound[j],'NaN',tmpcolvarsarray[0:npoints[i],whichcv[j]+1])
     colvarsarray[i]=np.ma.masked_invalid(colvarsarray[i])
     colvarsarray[i]=np.ma.mask_rows(colvarsarray[i])
     gradv[i]=np.ma.masked_invalid(gradv[i])
     gradv[i]=np.ma.mask_rows(gradv[i])

  #Debug
  #for i in range (0,npoints[i]):
  #   print colvarsarray[0][i,0],gradv[0][i,0],gradv[0][i,1] 
  #sys.exit()     


# create masked colvarsarray and gradv filtering values according to the difference between the calculated Gaussian energy
# the one repoted in the COLVAR file

if do_gefilter: 
  if do_hills_bias or read_gfile:
    for i in range (0,ncolvars):
       if nactive[i]>0: 
         tmpcolvarsarray=colvarsarray[i]
         diffc=np.abs(tmpcolvarsarray[:,colgenerread[i]]-gaussenergy[i][:])     
         for j in range(0,ndim):
            gradv[i][0:npoints[i],j]=np.where(diffc[0:npoints[i]]>tgefilt,'NaN',gradv[i][0:npoints[i],j])
         colvarsarray[i][0:npoints[i],0]=np.where(diffc[0:npoints[i]]>tgefilt,'NaN',colvarsarray[i][0:npoints[i],0])
         colvarsarray[i]=np.ma.masked_invalid(colvarsarray[i])
         colvarsarray[i]=np.ma.mask_rows(colvarsarray[i])
         gradv[i]=np.ma.masked_invalid(gradv[i])
         gradv[i]=np.ma.mask_rows(gradv[i])
  #Debug
  #       for j in range (0,npoints[i]):
  #          print colvarsarray[i][j,0],colvarsarray[i][j,1],colvarsarray[i][j,2],gradv[i][j,0],gradv[i][j,1],colgenerread[i],tmpcolvarsarray[j,colgenerread[i]],gaussenergy[i][j] 
  #     else:
  #       for j in range (0,npoints[i]):
  #          print colvarsarray[i][j,0],colvarsarray[i][j,1],colvarsarray[i][j,2],gradv[i][j,0],gradv[i][j,1],colgenerread[i],colvarsarray[i][j,colgenerread[i]],gaussenergy[i][j]           
  #  sys.exit()     

for i in range (0,ncolvars):
   print "NUMBER OF POINTS FOR WALKER ",i,": ",len(colvarsarray[i][:,0][~colvarsarray[i][:,0].mask])

# CALCULATE NUMBER OF EFFECTIVE POINTS IF REQUIRED

if calc_epoints:
  print ("Calculating effective points...")
  for k in range (0,ncolvars):
     ntotpoints=len(colvarsarray[k][:,0][~colvarsarray[k][:,0].mask])
     if k==0:    
       colvarseff=[np.zeros((ntotpoints,ndim))]
       numinpoint=[np.ones((ntotpoints),dtype=np.int64)] 
       neffpoints=[1]
     else:
       colvarseff.append(np.zeros((ntotpoints,ndim)))
       numinpoint.append(np.ones((ntotpoints),dtype=np.int64))
       neffpoints.append(1)      
     arrayin=np.zeros((ntotpoints,ndim))
     for j in range (0,ndim):
        arrayin[:,j]=colvarsarray[k][:,whichcv[j]+1][~colvarsarray[k][:,whichcv[j]+1].mask]
     if do_fast_eff_p_calc: 
       colvarseff[k],neffpoints[k],numinpoint[k]=fast_calc_eff_points(ntotpoints, arrayin, numinpoint[k]) 
     else:
       if do_bin_eff_p_calc:
         colvarseff[k],neffpoints[k],numinpoint[k]=calc_eff_points_bin(ntotpoints, arrayin, numinpoint[k])
       else:
         colvarseff[k],neffpoints[k],numinpoint[k]=calc_eff_points(ntotpoints, arrayin, numinpoint[k])
       
     # MERGE POINTS
     if k==0:
       colvareffarray=colvarseff[k][0:neffpoints[k],:]
       numinarray=numinpoint[k][0:neffpoints[k]]
     else:
       colvareffarray=np.concatenate((colvareffarray,colvarseff[k][0:neffpoints[k],:]),axis=0)
       numinarray=np.concatenate((numinarray,numinpoint[k][0:neffpoints[k]]))
  
  # recalculate effective points
  nepoints=len(colvareffarray)
  colvarseff=np.zeros((nepoints,ndim))
  numinpoint=np.ones((nepoints),dtype=np.int64)
  if ncolvars==1:
    neffpoints=nepoints
    colvarseff=colvareffarray
    numinpoint=numinarray 
  if ncolvars>1:  
    neffpoints=1
    if do_fast_eff_p_calc:
      colvarseff,neffpoints,numinpoint=fast_calc_eff_points(nepoints, colvareffarray, numinarray)
    else:
      if do_bin_eff_p_calc:
        colvarseff,neffpoints,numinpoint=calc_eff_points_bin(nepoints, colvareffarray, numinarray) 
      else:
        colvarseff,neffpoints,numinpoint=calc_eff_points(nepoints, colvareffarray, numinarray)

  for i in range (0,neffpoints):
     with open(eff_points_file, 'a') as f:
         f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         f.write("%s \n" % (numinpoint[i]))
 
if read_efile:
  print ("Reading effective points from external file...")
  for n in range(0,nefiles):
     try:
         tryarray = np.loadtxt(efile[n])
         if n==0:
           colvareffarray=tryarray
         else:          
           colvareffarray=np.concatenate((colvareffarray,tryarray),axis=0)
     except IOError:
         pass
  nepoints=len(colvareffarray)
  colvarseff=np.zeros((nepoints,ndim))
  neffpoints=1
  arrayin=np.zeros((nepoints,ndim))
  arrayin[0:nepoints,0:ndim]=colvareffarray[0:nepoints:,1:ndim+1]
  numinpoint=colvareffarray[0:nepoints:,ndim+1]
  if do_fast_eff_p_calc:
    colvarseff,neffpoints, numinpoint=fast_calc_eff_points(nepoints, arrayin, numinpoint) 
  else:
    if do_bin_eff_p_calc:
      colvarseff,neffpoints, numinpoint=calc_eff_points_bin(nepoints, arrayin, numinpoint) 
    else:
      colvarseff,neffpoints, numinpoint=calc_eff_points(nepoints, arrayin, numinpoint)

# CALC FORCE ON EFFECTIVE POINTS

if calc_force_eff:
  print ("Calculating forces on effective points...")
  grad=np.zeros((neffpoints,ndim))
  weighttot=np.zeros((neffpoints,ndim))
  gradr=np.zeros((neffpoints,ndim))
  weightr=np.zeros((neffpoints,ndim)) 
  for k in range (0,ncolvars):
     ntotpoints=len(colvarsarray[k][:,0][~colvarsarray[k][:,0].mask])
     arrayin=np.zeros((ntotpoints,ndim))
     gradvin=np.zeros((ntotpoints,ndim)) 
     for j in range (0,ndim):
        arrayin[:,j]=colvarsarray[k][:,whichcv[j]+1][~colvarsarray[k][:,whichcv[j]+1].mask]
        gradvin[:,j]=gradv[k][:,j][~gradv[k][:,j].mask]
     gradr, weightr=calc_vhar_force(neffpoints, ntotpoints, colvarseff, arrayin, gradvin) 
     weighttot=weighttot+weightr 
     grad=grad+gradr*weightr
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints,j]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints,j]),0)
  for i in range(0,neffpoints):
     with open(force_points_file, 'a') as f:  
         #f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         for j in range (0,ndim-1):
           f.write("%s " % (weighttot[i,j]))
         f.write("%s \n " % (weighttot[i,ndim-1]))

# READ FORCE AND EFFECTIVE POINTS FROM EXTERNAL FILE

if read_ffile:
  print ("Reading effective points and forces from external file...")
  if nffiles>1:
    print ("Reading forces arising from multiple simulations.")
    print ("Note that each file must correspond to forces calculated on the same points,")
    print ("thereby the files must have the same length.")
  for n in range(0,nffiles):
     try:
         tryarray = np.loadtxt(ffile[n])
         if n==0:
           colvarseff=tryarray[:,0:ndim]
           gradr=tryarray[:,ndim:2*ndim]
           weightr=tryarray[:,2*ndim:3*ndim]
           neffpoints=len(colvarseff)
           weighttot=weightr
           grad=np.zeros((neffpoints,ndim)) 
           grad=grad+gradr*weightr
         else:
           if len(tryarray)!=neffpoints:
             print ("ERROR, please provide files with the same lenght")
             sys.exit()
           distance=np.amax(4*(colvarseff[:,:]-tryarray[:,0:ndim])*(colvarseff[:,:]-tryarray[:,0:ndim])/(width[0:ndim]*width[0:ndim])) 
           if distance>1:
             print ("ERROR, points where forces have been calculated are different")
             sys.exit()
           weightr=tryarray[:,2*ndim:3*ndim]
           weighttot=weighttot+weightr 
           gradr=tryarray[:,ndim:2*ndim]
           grad=grad+gradr*weightr
     except IOError:
         pass
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints,j]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints,j]),0)
  for i in range(0,neffpoints):
     with open(force_points_file_comb, 'a') as f:
         #f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         for j in range (0,ndim-1):
            f.write("%s " % (weighttot[i,j]))
         f.write("%s \n " % (weighttot[i,ndim-1]))

if do_bin_data:
  bincolvars=np.zeros((neffpoints,ndim))         
  bingrad=np.zeros((neffpoints,ndim))
  binweight=np.zeros((neffpoints,ndim))
  binnumbers=0
  if do_fast_bin_data: 
    bincolvars, bingrad, binweight, binnumbers=fast_bin_data(neffpoints, colvarseff, weighttot, grad) 
  else:
    bincolvars, bingrad, binweight, binnumbers=bin_data(neffpoints, colvarseff, weighttot, grad)

  for i in range(0,binnumbers):
     with open(force_bin_file, 'a') as f:
         for j in range (0,ndim):
            f.write("%s " % (bincolvars[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (bingrad[i,j]))
         for j in range (0,ndim-1): 
            f.write("%s " % (binweight[i,j]))
         f.write("%s \n " % (binweight[i,ndim-1]))

print("--- %s seconds ---" % (time.time() - start_time)) 
