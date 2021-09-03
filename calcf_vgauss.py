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
    parser.add_argument("-temp", "--temp", help="Temperature (default is in Kelvin)", \
                        default=-1.0,type=float, required=False)
    parser.add_argument("-skip", "--skip", help="skip points when calculating forces (default 1)", \
                        default=1,type=int, required=False)
    parser.add_argument("-trfr1", "--trajfraction1", help="starting frame fraction of the trajectory (range 0-1) to be used for force calculation (default is 0)", \
                        default=0.0,type=float, required=False)
    parser.add_argument("-trfr2", "--trajfraction2", help="last frame fraction of the trajectory (range 0-1) to be used for force calculation (default is 1)", \
                        default=1.0,type=float, required=False)
    parser.add_argument("-kb", "--kb", help="Boltzmann factor to define free energy units. Default is 0.00831... kJ/mol", \
                        default=0.00831446261815324,type=float, required=False)
    parser.add_argument("-wf", "--widthfactor", help="Scaling factor of the width (wfact) to assign the force constant (k=kb*temp*(wfact*wfact)/(width*width); default is 1 (width is read in the GRID defined in the input file)", \
                        default=1.0,type=float, required=False)
    parser.add_argument("-colvarbias_column", "--read_colvarbias_column", help="read biasing force from COLVAR_FILE at a specified number of columns after the associated CV (e.g. 1 is right after the CV)", \
                        default=-1,type=int, required=False)
    parser.add_argument("-colvars","--colvars", \
                        help="Use default parameters for Colvars", \
                        default=False, dest='do_colv', action='store_true')
    parser.add_argument("-internalf","--internalforces", \
                        help="Provided free energy gradients are based on internal forces", \
                        default=False, dest='do_intern', action='store_true')
    parser.add_argument("-calcmetaf","--calcmetabiasforce", \
                        help="Calculate biasing forces of metadynamics from a HILLS file which stores the deposited Gaussian hills (TIME CV1 CV2... SIGMA_CV1 SIGMA_CV2... HEIGHT). By default metadynamics bias calculation is OFF", \
                        default=False, dest='do_hbias', action='store_true')
    parser.add_argument("-noforce","--nocalcforce", \
                        help="Do not calculate forces. By default forces calculation is ON", \
                        default=True, dest='do_force', action='store_false')
    parser.add_argument("-obgf", "--outbiasgradfile", \
                        help="output file of bias gradients for each frame", \
                        default="bias_grad.out",type=str, required=False)
    parser.add_argument("-oepf", "--outeffpointsfile", \
                        help="output file of effective points to calculate forces", \
                        default="eff_points.out",type=str, required=False)    
    parser.add_argument("-oeff", "--outeffforcefile", \
                        help="output file effective points and forces", \
                        default="force_on_eff_points.out",type=str, required=False)
    parser.add_argument("-oeff1", "--outeffforcefile1", \
                        help="output file effective points and forces using first half of the trajectory", \
                        default="force_on_eff_points1.out",type=str, required=False)
    parser.add_argument("-oeff2", "--outeffforcefile2", \
                        help="output file effective points and forces using second half of the trajectory", \
                        default="force_on_eff_points2.out",type=str, required=False)
    parser.add_argument("-ocmbeff", "--outcombeffforcefile", \
                        help="output combined file effective points and forces", \
                        default="force_on_eff_points_comb.out",type=str, required=False)
    parser.add_argument("-obff", "--outbinforcefile", \
                        help="output file of binned colvar and forces", \
                        default="force_on_bin_points.out",type=str, required=False)
    parser.add_argument("-olf", "--outlabelfile", \
                        help="output file of labels (assigned bins along colvar) ", \
                        default="label.out",type=str, required=False)
    parser.add_argument("-nobdat","--nobindata", \
                        help="Do not bin data according to provided grid", \
                        default=True, dest='do_bdat', action='store_false')
    parser.add_argument("-nobound","--noboundaries", \
                        help="Do not exclude data beyond grid boundaries", \
                        default=True, dest='do_bound', action='store_false')
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
    parser.add_argument("-label","--label", \
                        help="label COLVARS according to the effective foints", \
                        default=False, dest='do_label', action='store_true')
    parser.add_argument("-wrtlabelscrd","---writelabelscoord", \
                        help="write COLVARS for each label", \
                        default=False, dest='write_label_coord', action='store_true')
    parser.add_argument("-hlfl","--hillfreqlarge", \
                        help="Metadynamics in which HILLS are stored more frequently than COLVARS", \
                        default=False, dest='do_hlfl', action='store_true')
    parser.add_argument("-gefilt","--gaussianenergyfilter", \
                        help="Filter data by comparing the calculated gaussian energy with the one reported in the COLVAR file according to -valgefilt", \
                        default=False, dest='do_gefilt', action='store_true')
    parser.add_argument("-colgener", "--colgener", help="Column to read the Gaussian energy in the COLVAR file for filtering (Default is the second last)", \
                        default=-1,type=int, required=False)
    parser.add_argument("-valgefilt", "--valgefilt", help="Difference threshold between calculated gaussian energy and the one reported in the COLVAR file for filtering ", \
                        default=1000.0,type=float, required=False)
    parser.add_argument("-nfr", "--numframerest", help="Number of frames to filter out before and after restart ", \
                        default=-1,type=int, required=False)
    parser.add_argument("-backres","--backrestart", \
                        help="Filter out just data before a restart according to --numframerest", \
                        default=False, dest='do_backres', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()

# Variables
ifile=args.inputfile
do_hills_bias=args.do_hbias
do_colvars=args.do_colv
do_internalf=args.do_intern
do_bin_data=args.do_bdat
do_bin_data=args.do_bdat
do_label=args.do_label
writelabelscoord=args.write_label_coord
do_force=args.do_force
do_boundaries=args.do_bound
do_gefilter=args.do_gefilt
do_just_eff_points=args.do_jceffp
do_just_hills_bias=args.do_jmetab
do_fast_eff_p_calc=args.do_feffpc
do_bin_eff_p_calc=args.do_effpb
do_fast_bin_data=args.do_fbind
do_backrestart=args.do_backres
bias_grad_file=args.outbiasgradfile
labelfile=args.outlabelfile
eff_points_file=args.outeffpointsfile
force_points_file=args.outeffforcefile
force_points_file_comb=args.outcombeffforcefile
force_points_file1=args.outeffforcefile1
force_points_file2=args.outeffforcefile2
force_bin_file=args.outbinforcefile
temp=args.temp
skip=args.skip
kb=args.kb
widthfact=args.widthfactor
colvarbias_column=args.read_colvarbias_column
tgefilt=args.valgefilt
nfrestart=args.numframerest
colgener=args.colgener
do_large_hfreq=args.do_hlfl
trajfraction1=args.trajfraction1
trajfraction2=args.trajfraction2

# parameters
if do_colvars:
  hcutoff=11.5 # cutoff for Gaussians
else:
  hcutoff=6.25 # cutoff for Gaussians
wcutoff=18.75 # cutoff for Gaussians in weight calculation

if trajfraction1>=1.0 or trajfraction1<0.0:
  print ("ERROR: please select a trajectory starting point between 0 and 1")  
  sys.exit()


if trajfraction2>1.0 or trajfraction2<=0.0:
  print ("ERROR: please select a trajectory last point between 0 and 1")   
  sys.exit()

if trajfraction2<=trajfraction1:
  print ("ERROR: last point of the trajectory to be used must be larger than the first")
  sys.exit()  

if do_bin_eff_p_calc==False:
  do_fast_eff_p_calc=False 

calc_epoints=True # True unless is read from input
calc_force_eff=True # True unless is read from input

ncolvars=0
ndim=0
ngfiles=0
nefiles=0
nffiles=0
nfcomp=0
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
      print ("ERROR, COLVAR_FILE must be specified just one time on a single line") 
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
          print ("ERROR, HILLS_FILE must be specified just one time on a single line")
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
            print ("ERROR, HILLS_CVS not specified or specified more than once on a single line")
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
    if str(parts[0])=="READ_APP_FORCE":
      if colvarbias_column>0:
         print ("ERROR: you are reading the biasing forces two times; from a file (through READ_APP_FORCE) and also from the COLVAR_FILE (through -colvarbias_column). Select just one of the two options. ")
         sys.exit()  
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
      pluto=np.zeros((ndim),dtype=np.int8)
      if nparts>2: 
        if str(parts[2])=="REMOVE_COMP":
          for i in range (3,nparts):
             pluto[int(parts[i])-1]=1
      if nffiles==0:
        rcvcomp=[pluto]
      if nffiles>0:
        rcvcomp.append(pluto)  
      nffiles=nffiles+1

print ("Input read")

if colvarbias_column>0:
  read_gfile=True
  do_hills_bias=False

if do_just_eff_points:
  calc_epoints=True
  do_hills_bias=False
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False  
  do_bin_data=False
  print ("Requested just derivation of effective points (e.g. GRID on which mean forces will be evaluated) and nothing else")

if do_just_hills_bias:
  calc_epoints=False
  do_hills_bias=True
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False
  do_bin_data=False
  print ("Requested just derivation applied forces from HILLS files for each simulation frame and nothing else")
  
if ncolvars==0:
  calc_epoints=False  
  calc_force_eff=False

if do_force==False:
  calc_force_eff=False  

if do_hills_bias:
  do_internalf=False

internalf=1.0
if do_internalf:
  internalf=0.0
  print ("The force for each frame read from file (through READ_APP_FORCE or -colvarbias_column) is the internal force: FCAM not applied, mean forces are calculated as the local average of the internal forces") 
  if colvarbias_column<=1:
    print ("ERROR: both total and applied force must be provided to evaluate the internal force; -colvarbias_column must be larger than 1")
    sys.exit()    

if read_gfile==False:
  if do_hills_bias==False:
    calc_force_eff=False
    print ("NOTE: no option for reading applied forces from files or calculating them through HILLS files was selected: mean forces on effective points will not be calculated")

if ndim==0:
  print ("ERROR: number of variables is zero, please provide some to continue")
  sys.exit()
       
if calc_force_eff and temp<0:
  print ("ERROR: temperature for calculating forces not provided or negative value")
  sys.exit()
 
if read_gfile:
  if colvarbias_column<0:
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
      f.write("# numeff, colvar, npoints \n")

if do_label:
  with open(labelfile, 'w') as f:
      f.write("# colvar, label, ntraj \n") 

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
         f.write("  %s  " % (width[j]))
         f.write("  %s  " % (npointsv[j]))
         f.write("  %s  \n" % periodic[j])
  with open(force_points_file1, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]))
         f.write("  %s  " % (npointsv[j]))
         f.write("  %s  \n" % periodic[j])
  with open(force_points_file2, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]))
         f.write("  %s  " % (npointsv[j]))
         f.write("  %s  \n" % periodic[j])

if read_ffile:
  with open(force_points_file_comb, 'w') as f:
      f.write("# %s \n" % ndim)
      for j in range (0,ndim):
         f.write("# %s " % lowbound[j])
         f.write("  %s  " % (width[j]))
         f.write("  %s  " % (npointsv[j]))
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
      diffc[0:neffp,:]=diffc[0:neffp,:]/width[0:ndim]
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
   mywidth=width
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
   mywidth=width
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

# DO LABELS: ASSIGN BIN ALONG A COLVAR FILE

def assign_bins(numpoints, colvars, numepoints, effparray):
   mywidth=width
   diffc=np.zeros((numepoints,ndim))
   totperiodic=np.sum(periodic[0:ndim])
   labelbin=np.zeros((numpoints),dtype=np.int64)
   for i in range(0,numpoints):
      diffc[:,:]=colvars[i,:]-effparray[:,:]
      if totperiodic>0: 
        diffc=diffc/box[0:ndim]
        diffc=diffc-np.rint(diffc)*periodic[0:ndim]
        diffc=diffc*box[0:ndim]
      diffc=diffc/mywidth[0:ndim]
      distance=np.amax(np.abs(diffc),axis=1)
      labelbin[i]=np.argmin(distance)
      mindistance=distance[labelbin[i]]
      if mindistance>0.5:
        labelbin[i]=-999
      if np.isnan(mindistance):
        labelbin[i]=-999
   return labelbin


#ROUTINE TO CALCULATE THE FORCE ON A SET OF POINTS

def calc_vhar_force(numepoints, numpoints, effparray, colvars, gradbias):
   grade=np.zeros((numepoints,ndim))
   grade1=np.zeros((numepoints,ndim)) #for error calculation
   grade2=np.zeros((numepoints,ndim)) #for error calculation
   diffc=np.zeros((numpoints,ndim))
   tweights=np.zeros((numepoints))
   tweights1=np.zeros((numepoints))
   tweights2=np.zeros((numepoints))
   totperiodic=np.sum(periodic[0:ndim]) 
   for i in range(0,numepoints):
      diffc=colvars[:,:]-effparray[i,:]
      if totperiodic>0:
        diffc=diffc/box[0:ndim]        
        diffc=diffc-np.rint(diffc)*periodic[0:ndim]
        diffc=diffc*box[0:ndim]
      diffc=widthfact*diffc/width[0:ndim]
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
        grade[i,:]=-np.sum((internalf*widthfact*kb*temp*diffc[whichpoints[0][:],:]/width[:]+gradbias[whichpoints[0][:],:])*weight[:,np.newaxis],axis=0) 
        tweights[i]=np.sum(weight)
        # fragment frames in two portions to calculate the error 
        weight1=np.where(np.cumsum(weight)<0.5*tweights[i],weight,0)
        weight2=np.where(np.cumsum(weight)>=0.5*tweights[i],weight,0)
        grade1[i,:]=-np.sum((internalf*widthfact*kb*temp*diffc[whichpoints[0][:],:]/width[:]+gradbias[whichpoints[0][:],:])*weight1[:,np.newaxis],axis=0) 
        grade2[i,:]=-np.sum((internalf*widthfact*kb*temp*diffc[whichpoints[0][:],:]/width[:]+gradbias[whichpoints[0][:],:])*weight2[:,np.newaxis],axis=0)
        tweights1[i]=np.sum(weight1)
        tweights2[i]=np.sum(weight2)
        if tweights[i]>0:
          grade[i,:]=grade[i,:]/tweights[i] 
        if tweights1[i]>0:
          grade1[i,:]=grade1[i,:]/tweights1[i] 
        if tweights2[i]>0:
          grade2[i,:]=grade2[i,:]/tweights2[i]    
   return grade,tweights,grade1,tweights1,grade2,tweights2

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
  print ("Reading applied forces from external file...") 
  if colvarbias_column<=0:
    if ngfiles==1: 
      print ("Reading unique external file with applied force for all COLVAR files ...")
      gradarray = [np.loadtxt(gfile[0])]
      if np.sum(npoints[:])!=len(gradarray[0]):
        print ("ERROR: gradient file doesn't match COLVAR files")
        sys.exit() 
      totpoints=0
      for k in range (0,ncolvars):
         if k==0:
           gradv=[gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1]]
           if do_gefilter:
             gaussenergy=[gradarray[0][totpoints:totpoints+npoints[k],ndim+1]]
         if k>0:
           gradv.append(gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1])
           if do_gefilter:
             gaussenergy.append(gradarray[0][totpoints:totpoints+npoints[k],ndim+1]) 
         totpoints=totpoints+npoints[k]
    else:
      print ("Reading separate external files with applied force for each COLVAR file ...") 
      for k in range (0,ngfiles):
         if k==0:
           gradarray = [np.loadtxt(gfile[k])]
           if npoints[k]!=len(gradarray[k]):
             print ("ERROR: gradient file doesn't match COLVAR file")
             sys.exit()
           gradv=[gradarray[k][:,1:ndim+1]]
           if do_gefilter:
             gaussenergy=[gradarray[k][:,ndim+1]]
         if k>0:
           gradarray.append(np.loadtxt(gfile[k]))
           gradv.append(gradarray[k][:,1:ndim+1])
           if do_gefilter:
             gaussenergy.append(gradarray[k][:,ndim+1])
  if colvarbias_column>0:
    totpoints=0
    for k in range (0,ncolvars):
       if k==0:
         for j in range (0,ndim):
            =colvarsarray[k][:,whichcv[j]+1]
         
# create masked colvarsarray and gradv eliminating frames before and after restart

if nfrestart>0:
  for i in range (0,ncolvars):
     getpoints=np.where(colvarsarray[i][1:npoints[i],0]<=colvarsarray[i][0:npoints[i]-1,0])
     getipoints=getpoints[0]-nfrestart+1
     if do_backrestart:
       getfpoints=getpoints[0]+1
     else:
       getfpoints=getpoints[0]+nfrestart+1
     for j in range (0,getpoints[0].size): 
        if getipoints[j]<0:
          getipoints[j]=getpoints[0][j]      
        colvarsarray[i][getipoints[j]:getfpoints[j],0]=np.nan
  #Debug
  #for i in range (0,npoints[i]):
  #   print colvarsarray[0][i,0],colvarsarray[0][i,1],colvarsarray[0][i,2],gradv[0][i,0],gradv[0][i,1] 
  #sys.exit()     

# create masked colvarsarray and gradv eliminating values beyond boundaries
if do_boundaries:
  for i in range (0,ncolvars):
     tmpcolvarsarray=colvarsarray[i]
     for j in range(0,ndim):
        gradv[i][0:npoints[i],j]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]<lowbound[j],np.nan,gradv[i][0:npoints[i],j])
        gradv[i][0:npoints[i],j]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]>upbound[j],np.nan,gradv[i][0:npoints[i],j])
        colvarsarray[i][0:npoints[i],whichcv[j]+1]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]<lowbound[j],np.nan,tmpcolvarsarray[0:npoints[i],whichcv[j]+1])
        colvarsarray[i][0:npoints[i],whichcv[j]+1]=np.where(tmpcolvarsarray[0:npoints[i],whichcv[j]+1]>upbound[j],np.nan,tmpcolvarsarray[0:npoints[i],whichcv[j]+1])
     colvarsarray[i]=np.ma.masked_invalid(colvarsarray[i])
     colvarsarray[i]=np.ma.mask_rows(colvarsarray[i])
     gradv[i]=np.ma.masked_invalid(gradv[i])
     gradv[i]=np.ma.mask_rows(gradv[i])

if nfrestart>0:
  for i in range (0,ncolvars):
     for j in range(0,ndim):
        gradv[i][:,j]=np.where(np.isnan(colvarsarray[i][:,0]),np.nan,gradv[i][:,j])
     colvarsarray[i]=np.ma.masked_invalid(colvarsarray[i])
     colvarsarray[i]=np.ma.mask_rows(colvarsarray[i])
     gradv[i]=np.ma.masked_invalid(gradv[i])
     gradv[i]=np.ma.mask_rows(gradv[i])
     
  #Debug
  #for i in range (0,npoints[i]):
  #   print colvarsarray[0][i,0],colvarsarray[0][i,1],colvarsarray[0][i,2],gradv[0][i,0],gradv[0][i,1] 
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
            gradv[i][0:npoints[i],j]=np.where(diffc[0:npoints[i]]>tgefilt,np.nan,gradv[i][0:npoints[i],j])
         colvarsarray[i][0:npoints[i],0]=np.where(diffc[0:npoints[i]]>tgefilt,np.nan,colvarsarray[i][0:npoints[i],0])
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
   print ("NUMBER OF POINTS FOR WALKER ",i,": ",len(colvarsarray[i][:,0][~colvarsarray[i][:,0].mask]))

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
  if do_label:
    neffpoints=nepoints
    colvarseff=arrayin
  else:  
    if do_fast_eff_p_calc:
      colvarseff,neffpoints, numinpoint=fast_calc_eff_points(nepoints, arrayin, numinpoint) 
    else:
      if do_bin_eff_p_calc:
        colvarseff,neffpoints, numinpoint=calc_eff_points_bin(nepoints, arrayin, numinpoint) 
      else:
        colvarseff,neffpoints, numinpoint=calc_eff_points(nepoints, arrayin, numinpoint)

# DEGUB check eff points

#  with open("check_eff_points.dat", 'w') as f:
#      for i in range (0,neffpoints):
#         f.write("%s " % (i))
#         for j in range (0,ndim):
#            f.write("%s " % (colvarseff[i,j]))
#         f.write("%s \n" % (numinpoint[i]))

# DEBUG  

# CALC FORCE ON EFFECTIVE POINTS

if calc_force_eff:
  print ("Calculating forces on effective points...")
  grad=np.zeros((neffpoints,ndim))
  weighttot=np.zeros((neffpoints))
  gradr=np.zeros((neffpoints,ndim))
  weightr=np.zeros((neffpoints))
  grad1=np.zeros((neffpoints,ndim))
  weighttot1=np.zeros((neffpoints))
  gradr1=np.zeros((neffpoints,ndim))
  weightr1=np.zeros((neffpoints))
  grad2=np.zeros((neffpoints,ndim))
  weighttot2=np.zeros((neffpoints))
  gradr2=np.zeros((neffpoints,ndim))
  weightr2=np.zeros((neffpoints)) 
  for k in range (0,ncolvars):
     initframe=int(trajfraction1*npoints[k])
     lastframe=int(trajfraction2*npoints[k])
     ntotpoints=len(colvarsarray[k][initframe:lastframe:skip,0][~colvarsarray[k][initframe:lastframe:skip,0].mask])
     arrayin=np.zeros((ntotpoints,ndim))
     gradvin=np.zeros((ntotpoints,ndim))
     for j in range (0,ndim):
        arrayin[:,j]=colvarsarray[k][initframe:lastframe:skip,whichcv[j]+1][~colvarsarray[k][initframe:lastframe:skip,whichcv[j]+1].mask]
        gradvin[:,j]=gradv[k][initframe:lastframe:skip,j][~gradv[k][initframe:lastframe:skip,j].mask]
     gradr, weightr, gradr1, weightr1, gradr2, weightr2=calc_vhar_force(neffpoints, ntotpoints, colvarseff, arrayin, gradvin) 
     weighttot=weighttot+weightr 
     weighttot1=weighttot1+weightr1
     weighttot2=weighttot2+weightr2
     for j in range(0,ndim):
        grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints]
        grad1[0:neffpoints,j]=grad1[0:neffpoints,j]+gradr1[0:neffpoints,j]*weightr1[0:neffpoints]
        grad2[0:neffpoints,j]=grad2[0:neffpoints,j]+gradr2[0:neffpoints,j]*weightr2[0:neffpoints]
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints]),0)
     grad1[0:neffpoints,j]=np.where(weighttot1[0:neffpoints]>0,grad1[0:neffpoints,j]/(weighttot1[0:neffpoints]),0)
     grad2[0:neffpoints,j]=np.where(weighttot2[0:neffpoints]>0,grad2[0:neffpoints,j]/(weighttot2[0:neffpoints]),0)
  for i in range(0,neffpoints):
     with open(force_points_file, 'a') as f:  
         #f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         f.write("%s \n " % (weighttot[i]))
  for i in range(0,neffpoints):
     with open(force_points_file1, 'a') as f:
         #f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad1[i,j]))
         f.write("%s \n " % (weighttot1[i]))
  for i in range(0,neffpoints):
     with open(force_points_file2, 'a') as f:
         #f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad2[i,j]))
         f.write("%s \n " % (weighttot2[i]))

# CALC LABELS

if do_label:
  for k in range (0,ncolvars):
     arrayin=np.zeros((npoints[k],ndim)) 
     binlabel=np.zeros((npoints[k]),dtype=np.int64)
     for j in range (0,ndim):
        arrayin[:,j]=colvarsarray[k][:,whichcv[j]+1]

# DEGUB check eff points

#     with open("colvar_check.dat", 'w') as f:
#         for i in range (0,npoints[k]):
#            for j in range (0,ndim-1):
#               f.write("%s " % (arrayin[i,j]))
#            f.write("%s \n" % (arrayin[i,ndim-1]))

# DEBUG  

     binlabel=assign_bins(npoints[k], arrayin, neffpoints, colvarseff)
     with open(labelfile, 'a') as f:
         if writelabelscoord:
           for i in range(0,npoints[k]):
              if binlabel[i]!=-999:
                if np.ma.is_masked(colvarsarray[k][i,0]):
                  f.write("%s " % (-999)) 
                  for j in range (0,ndim):
                     f.write("%s " % (-999))
                  f.write("%s %s \n " % (-999,k))
                else:
                  f.write("%s " % (colvarsarray[k][i,0]))
                  for j in range (0,ndim): 
                     f.write("%s " % (colvarseff[binlabel[i],j])) 
                  f.write("%s %s \n " % (binlabel[i],k))
              else:
                f.write("%s " % (-999)) 
                for j in range (0,ndim):
                   f.write("%s " % (-999))
                f.write("%s %s \n " % (binlabel[i],k))
         else:
           for i in range(0,npoints[k]):
              if binlabel[i]!=-999:
                if np.ma.is_masked(colvarsarray[k][i,0]):
                  f.write("%s " % (-999))
                  f.write("%s %s \n " % (-999,k))
                else:
                  f.write("%s " % (colvarsarray[k][i,0]))
                  f.write("%s %s \n " % (binlabel[i],k))
              else:
                f.write("%s " % (-999))
                f.write("%s %s \n " % (binlabel[i],k))


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
         weightr=np.zeros((len(tryarray),ndim))
         ncol=np.ma.size(tryarray,axis=1)
         if n==0:
           colvarseff=tryarray[:,0:ndim]
           gradr=tryarray[:,ndim:2*ndim]
           if ncol>2*ndim+1:
             weightr=tryarray[:,2*ndim:3*ndim]
           else:
             if ncol>2*ndim:
               for j in range(0,ndim):
                  weightr[:,j]=tryarray[:,2*ndim]
           #weightr=tryarray[:,2*ndim]
           # REMOVE BIAS COMPONENTS IF REQUESTED 
           for j in range(0,ndim):
              if rcvcomp[n][j]==1:
                weightr[:,j]=np.zeros((len(tryarray)))           
           neffpoints=len(colvarseff)
           weighttot=weightr
           grad=np.zeros((neffpoints,ndim)) 
           for j in range(0,ndim):
              grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints,j]
         else:
           if len(tryarray)!=neffpoints:
             print ("ERROR, please provide files with the same lenght")
             sys.exit()
           distance=np.amax((colvarseff[:,:]-tryarray[:,0:ndim])*(colvarseff[:,:]-tryarray[:,0:ndim])/(width[0:ndim]*width[0:ndim])) 
           if distance>1:
             print ("ERROR, points where forces have been calculated are different")
             sys.exit()
           if ncol>2*ndim+1:
             weightr=tryarray[:,2*ndim:3*ndim]
           else:
             if ncol>2*ndim:
               for j in range(0,ndim):
                  weightr[:,j]=tryarray[:,2*ndim]   
           #weightr=tryarray[:,2*ndim]
           # REMOVE BIAS COMPONENTS IF REQUESTED 
           for j in range(0,ndim):
              if rcvcomp[n][j]==1:
                weightr[:,j]=np.zeros((len(tryarray)))     
           weighttot=weighttot+weightr 
           gradr=tryarray[:,ndim:2*ndim]
           for j in range(0,ndim):
              grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints,j]
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

print("--- %s seconds ---" % (time.time() - start_time)) 

