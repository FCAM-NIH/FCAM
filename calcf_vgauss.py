import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--inputfile", \
                        help="input file for analysis", \
                        type=str, required=True)
    parser.add_argument("-temp", "--temp", help="Temperature (in Kelvin) for calculating the force constant (k) according to half bin width (k=4*kb*temp/(width*width)) ", \
                        default=-1.0,type=float, required=False)
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
    parser.add_argument("-obff", "--outbinforcefile", \
                        help="output file of binned colvar and forces", \
                        default="force_on_bin_points.out",type=str, required=False)
    parser.add_argument("-nobdat","--nobindata", \
                        help="Do not bin data according to provided grid", \
                        default=True, dest='do_bdat', action='store_false')
    parser.add_argument("-jceffp","--justcalceffpoints", \
                        help="Calculate effective points and do nothing else. COLVARS and GRID data must at least be provided", \
                        default=False, dest='do_jceffp', action='store_true')
    parser.add_argument("-jcmetab","--justcalcmetabias", \
                        help="Calculate metadynamics bias potential and do nothing else. COLVARS, HILLS and GRID data must at least be provided", \
                        default=False, dest='do_jmetab', action='store_true')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

args = parse()


# parameters
hcutoff=6.25 # cutoff for Gaussians
kb=0.00831446261815324

# Variables
ifile=args.inputfile
do_hills_bias=args.do_hbias
do_bin_data=args.do_bdat
do_just_eff_points=args.do_jceffp
do_just_hills_bias=args.do_jmetab
bias_grad_file=args.outbiasgradfile
eff_points_file=args.outeffpointsfile
force_points_file=args.outeffforcefile
force_bin_file=args.outbinforcefile
temp=args.temp

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
      print ("Error, COLVAR_FILE must be specidied just one time on a single line") 
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
          print ("Error, HILLS_FILE must be specidied just one time on a single line")
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
            print ("Error, HILLS_CVS not specified or specidied more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break  
             nact=nact+1
          nactive=[nact]
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int32)
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
          print ("Error, HILLS_FILE must be specidied just one time on a single line")
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
            print ("Error, HILLS_CVS not specified or specidied more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE":
               break 
             nact=nact+1
          nactive.append(nact)
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int32)
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
        if float(parts[1])>=float(parts[3]):
          print ("Error: lower boundary must be smaller than upper boudary ") 
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
          print ("Error: lower boundary must be smaller than upper boudary ")
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
  print ("Error: number of variables is zero, please provide some to continue")
  sys.exit()
       
if calc_force_eff and temp<0:
  print ("Error: temperature for calculating forces not provided or negative value")
  sys.exit()
 
if read_gfile:
  if ngfiles!=1:
    if ngfiles!=ncolvars:
      print ("Error: please provide a unique gradient file")
      print ("or a gradient file for each colvar.")
      print ("Note that in either case this must be consistent with the ORDERED set of colvar files provided.")
      sys.exit()

if do_hills_bias:
  with open(bias_grad_file, 'w') as f:
      f.write("# Time, grad,Gaussenergy,rep \n")

if calc_epoints:
  with open(eff_points_file, 'w') as f:
      f.write("# numeff, colvar, rep \n")

if calc_force_eff:
  with open(force_points_file, 'w') as f:
      f.write("# numeff, colvar, gradient \n")

has_hills=np.array(has_hills)
nactive=np.array(nactive)
upbound=np.array(upbound)
lowbound=np.array(lowbound)
npointsv=np.array(npointsv)
periodic=np.array(periodic,dtype=np.int8)

box=upbound-lowbound
width=box/npointsv

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

if len(cunique)!=len(cfile):
  print ("Error: same COLVAR file introduced multiple times in the input")
  sys.exit()

if len(hunique)!=len(hfile):
  print ("Error: same HILLS file introduced multiple times in the input")
  sys.exit()

if len(vunique)!=len(whichcv):
  print ("Error: same CV introduced multiple times in the input")
  sys.exit()

iactive=np.zeros((ncolvars,ndim),dtype=np.int8)
iactive[:,:]=-1

allfound=True
for i in range (0,ncolvars):
   if nactive[i]>ndim:
     print ("Error: number of HILLS_CVS larger than total number of CVS")
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
  print ("Error: HILLS_CVS must be part of the CVS used for CV")
  sys.exit()

# read the hills files
for i in range (0,ncolvars):
   if has_hills[i]:
     if i==0:
       hillsarray=[np.loadtxt(hfile[i])]
       nhills=[len(hillsarray[i])]
       #tmp_cvhills=[hillsarray[i][:,1:nactive[i]+1]]
       #tmp_deltahills=[hillsarray[i][:,nactive[i]+1:2*nactive[i]+1]]
       #whills=[hillsarray[i][:,2*nactive[i]+1]]
   
     else:

       hillsarray.append(np.loadtxt(hfile[i]))
       nhills.append(len(hillsarray[i]))
       #tmp_cvhills.append(hillsarray[i][:,1:nactive[i]+1])
       #tmp_deltahills.append(hillsarray[i][:,nactive[i]+1:2*nactive[i]+1])
       #whills.append(hillsarray[i][:,2*nactive[i]+1])

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
       #tmp_cvhills.append(0)
       #tmp_deltahills.append(0)
       #whills.append(0)

#nhillsmax=np.amax(nhills[:])

#diff=[np.zeros((nhillsmax,ndim))]
#diff2=[np.zeros((nhillsmax,ndim))]
#expdiff=[np.zeros((nhillsmax))]

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

# ROUTINE TO CALCULATE THE NUMBER OF EFFECTIVE POINTS (USEFUL TO REDUCE NUMBER OF POINTS AND THUS FORCE CALCULATIONS)

def calc_eff_points(numpoints, inputarray):
   diffc=np.zeros((numpoints,ndim))
   distance=np.zeros((numpoints))
   neffp=1 
   ceff=np.zeros((numpoints,ndim))
   ceff[0,:]=inputarray[0,:]
   for i in range(1,numpoints):
      diffc[0:neffp,:]=ceff[0:neffp,:]-inputarray[i,:]
      diffc[0:neffp,:]=diffc[0:neffp,:]/box[0:ndim]
      diffc[0:neffp,:]=diffc[0:neffp,:]-np.rint(diffc[0:neffp,:])*periodic[0:ndim]
      diffc[0:neffp,:]=diffc[0:neffp,:]*box[0:ndim]
      diffc[0:neffp,:]=2.0*diffc[0:neffp,:]/width[0:ndim]
      diffc[0:neffp,:]=diffc[0:neffp,:]*diffc[0:neffp,:]
      distance[0:neffp]=np.sum(diffc[0:neffp,:],axis=1)
      mindistance=np.amin(distance[0:neffp])
      if mindistance>1:
        ceff[neffp,:]=inputarray[i,:]
        neffp=neffp+1
   return ceff[0:neffp], neffp

#ROUTINE TO CALCULATE THE FORCE ON A SET OF POINTS

def calc_vhar_force(numepoints, numpoints, effparray, colvars, gradbias):
   grade=np.zeros((numepoints,ndim))
   diffc=np.zeros((numpoints,ndim))
   tweights=np.zeros((numepoints)) 
   for i in range(0,numepoints):
      diffc=colvars[:,:]-effparray[i,:]
      diffc=diffc/box[0:ndim]        
      diffc=diffc-np.rint(diffc)*periodic[0:ndim]
      diffc=diffc*box[0:ndim]
      diffc=2.0*diffc/width[0:ndim]
      for j in range(0,ndim):
         weight=np.exp(np.sum(-0.5*diffc[0:numpoints,:]*diffc[0:numpoints,:],axis=1))
         if np.amax(weight)>0: 
           grade[i,j]=-np.average(2.0*kb*temp*diffc[0:numpoints,j]/width[j]+gradbias[0:numpoints,j],weights=weight)
           tweights[i]=np.sum(weight) 
   return grade,tweights

# ROUTINE TO BIN THE DATA STARTING FROM POINT WITH LARGEST WEIGHT

def bin_data(numepoints, effparray, weights, gradarray):
   colvarsbineff=np.zeros((numepoints, ndim))
   nbins=1
   diffc=np.zeros((numepoints,ndim)) 
   distance=np.zeros((numepoints))
   gradbin=np.zeros((numepoints,ndim))
   weightbin=np.zeros((numepoints))
   numinbin=np.zeros((numepoints),dtype=np.int32)
   indexmax=np.argmax(weights)
   colvarsbineff[0,:]=effparray[indexmax,:]
   for i in range(0,numepoints):
      diffc[i,:]=effparray[i,:]-effparray[indexmax,:]
      diffc[i,:]=diffc[i,:]/box[0:ndim]
      diffc[i,:]=diffc[i,:]-np.rint(diffc[i,:])*periodic[0:ndim]
      diffc[i,:]=diffc[i,:]*box[0:ndim]
      colvarbin=0.5*width+width*np.floor(diffc[i,:]/width)+effparray[indexmax,:]
      colvarbin=(colvarbin-(lowbound+0.5*box))/box[0:ndim]
      colvarbin=colvarbin-np.rint(colvarbin)*periodic[0:ndim]
      colvarbin=colvarbin*box[0:ndim]+(lowbound+0.5*box)
      diffc[0:nbins,:]=colvarsbineff[0:nbins,:]-colvarbin[:]
      diffc[0:nbins,:]=diffc[0:nbins,:]/box[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]-np.rint(diffc[0:nbins,:])*periodic[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]*box[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]/width[0:ndim]
      diffc[0:nbins,:]=diffc[0:nbins,:]*diffc[0:nbins,:]
      distance[0:nbins]=np.sum(diffc[0:nbins,:],axis=1)
      mindistance=np.amin(distance[0:nbins])  
      if mindistance>0.5:
        colvarsbineff[nbins,:]=colvarbin
        nbins=nbins+1
      whichbin=np.argmin(distance[0:nbins])
      numinbin[whichbin]=numinbin[whichbin]+1
      gradbin[whichbin,:]=gradbin[whichbin,:]+gradarray[i,:]
      weightbin[whichbin]=weightbin[whichbin]+weights[i]   
   for j in range(0,ndim):
      gradbin[0:nbins,j]=np.where(numinbin[0:nbins]>0,gradbin[0:nbins,j]/numinbin[0:nbins],0)
   weightbin[0:nbins]=np.where(numinbin[0:nbins]>0,weightbin[0:nbins]/numinbin[0:nbins],0)
   return colvarsbineff, gradbin, weightbin, nbins
 
# calc HILLS forces

if do_hills_bias:
  print ("Calculating metadynamics bias forces on each COLVAR point of the selected variables from the HILLS files...")
  for k in range (0,ncolvars):
     diff=np.zeros((nhills[k],nactive[k]))
     diff2=np.zeros((nhills[k],nactive[k]))
     expdiff=np.zeros((nhills[k]))
     whichhills=np.zeros((nhills[k]),dtype=np.int32)
     trh=0
     index=0
     dvec=np.arange(nhills[k])
     if nactive[k]>0:
       countinter=0 
       for i in range(0,npoints[k]):
          gaussenergy=0
          whichhills_old=whichhills
          if i>0 and colvarsarray[k][i,0]<colvarsarray[k][i-1,0]:
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
            gaussenergy=np.sum(expdiff[0:numhills])
            for j in range(0,nactive[k]):  
               gradv[k][i,iactive[k,j]]=np.sum(diff[0:numhills,j]*expdiff[0:numhills]/hillsarray[k][0:numhills,nactive[k]+1+j],axis=0)
          with open(bias_grad_file, 'a') as f:
              f.write("%s " % (colvarsarray[k][i,0]))
              for j in range(0,ndim):
                 f.write("%s " % (gradv[k][i,j]))
              f.write("%s " % gaussenergy)
              f.write("%s " % numhills)
              f.write("%s " % trh)
              f.write("%s " % index)
              f.write("%s \n" % (k))
     else:
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
    #if np.sum(npoints[:])!=len(gradarray[0]):
    #  print ("ERROR: gradient file doen't match COLVAR files") 
    totpoints=0
    for k in range (0,ncolvars):
       if k==0:
         gradv=[gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1]]
       if k>0:
         gradv.append(gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1])
       totpoints=totpoints+npoints[k]
  else:
    for k in range (0,ngfiles):
       if k==0:    
         gradv=[gradarray[:,1:ndim+1]]
       if k>0:
         gradv.append(gradarray[:,1:ndim+1]) 

# CALCULATE NUMBER OF EFFECTIVE POINTS IF REQUIRED

if calc_epoints:
  print ("Calculating effective points...")
  for k in range (0,ncolvars):
     if k==0:
       colvarseff=[np.zeros((npoints[k],ndim))] 
       neffpoints=[1]
     else:
       colvarseff.append(np.zeros((npoints[k],ndim)))
       neffpoints.append(1)
     arrayin=np.zeros((npoints[k],ndim))
     for j in range (0,ndim):
        arrayin[0:npoints[k],j]=colvarsarray[k][0:npoints[k],whichcv[j]+1] 
     colvarseff[k],neffpoints[k]=calc_eff_points(npoints[k], arrayin) 
 
     # MERGE POINTS
     if k==0:
       colvareffarray=colvarseff[k][0:neffpoints[k],:]
     else:
       colvareffarray=np.concatenate((colvareffarray,colvarseff[k][0:neffpoints[k],:]),axis=0)

  # recalculate effective points
  nepoints=len(colvareffarray)
  colvarseff=np.zeros((nepoints,ndim)) 
  if ncolvars>1:  
    neffpoints=1
    colvarseff,neffpoints=calc_eff_points(nepoints, colvareffarray)

  for i in range (0,neffpoints):
     with open(eff_points_file, 'a') as f:
         f.write("%s " % (i))
         for j in range (0,ndim-1):
            f.write("%s " % (colvarseff[i,j]))
         f.write("%s \n" % (colvarseff[i,ndim-1]))
 
if read_efile:
  print ("Reading effective points from external file...")
  for n in range(1,nefiles):
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
  colvarseff,neffpoints=calc_eff_points(nepoints, colvareffarray[:,1:ndim+1]) 

# CALC FORCE ON EFFECTIVE POINTS

if calc_force_eff:
  print ("Calculating forces on effective points...")
  grad=np.zeros((neffpoints,ndim))
  weighttot=np.zeros((neffpoints))
  gradr=np.zeros((neffpoints,ndim))
  weightr=np.zeros((neffpoints)) 
  for k in range (0,ncolvars):
     arrayin=np.zeros((npoints[k],ndim))
     for j in range (0,ndim):
        arrayin[0:npoints[k],j]=colvarsarray[k][0:npoints[k],whichcv[j]+1]
     gradr, weightr=calc_vhar_force(neffpoints, npoints[k], colvarseff, arrayin, gradv[k])  
     weighttot=weighttot+weightr 
     for j in range(0,ndim):
        grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints]
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints]),0)
  for i in range(0,neffpoints):
     with open(force_points_file, 'a') as f:  
         f.write("%s " % (i))
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         f.write("%s \n " % (weighttot[i]))

# READ FORCE AND EFFECTIVE POINTS FROM EXTERNAL FILE

if read_ffile:
  print ("Reading effective points and forces from external file...")
  if nffiles>1:
    print ("Reading forces arising from multiple simulations.")
    print ("Note that each file must correspond to forces calculated on the same points,")
    print ("thereby the files must have the same length.")
  for n in range(1,nffiles):
     try:
         tryarray = np.loadtxt(ffile[n])
         if n==0:
           colvarseff=tryarray[:,1:ndim+1]
           gradr=tryarray[:,ndim+1:2*ndim+1]
           weightr=tryarray[:,2*ndim+1]
           neffpoints=len(colvarseff)
           weighttot=weightr
           grad=gradr
         else:
           if len(tryarray)!=neffpoints:
             print ("Error, please provide files with the same lenght")
             sys.exit()
           distance=np.amax(4*(colvarseff[:,:]-tryarray[:,1:ndim+1])*(colvarseff[:,:]-tryarray[:,1:ndim+1])/(width[0:ndim]*width[0:ndim])) 
           if distance>1:
             print ("Error, points where forces have been calculated are different")
             sys.exit()
           weighttot=weighttot+weightr 
           for j in range(0,ndim):
              grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints]
     except IOError:
         pass
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints]),0)

if do_bin_data:
  bincolvars=np.zeros((neffpoints,ndim))         
  bingrad=np.zeros((neffpoints,ndim))
  binweight=np.zeros((neffpoints))
  binnumbers=0 
  bincolvars, bingrad, binweight, binnumbers=bin_data(neffpoints, colvarseff, weighttot, grad) 

  for i in range(0,binnumbers):
     with open(force_bin_file, 'a') as f:
         for j in range (0,ndim):
            f.write("%s " % (bincolvars[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (bingrad[i,j]))
         f.write("%s \n " % (binweight[i]))
 
