#!/usr/bin/env python

import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy
from errors import FCAM_Error
from numba import jit
import time
start_time = time.time()

# This is a python program to calculate mean forces based on the Force-Correction Analysis Method. 
# If you use this code please cite:
#
# Marinelli, F. and J.D. Faraldo-Gomez, Force-Correction Analysis Method for Derivation of Multidimensional Free-Energy Landscapes from Adaptively Biased Replica Simulations. J Chem Theory Comput, 2021, 17: p. 6775-6788
#
# Manual and tutorial can be downloaded from https://github.com/FCAM-NIH/FCAM
#

### Argparser ###
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-if", "--inputfile", \
                        help="input file for analysis", \
                        type=str, required=True)
    parser.add_argument("-temp", "--temp", help="Temperature (required)", \
                        default=-1.0,type=float, required=False)
    parser.add_argument("-skip", "--skip", help="skip points when calculating forces (default 1)", \
                        default=1,type=int, required=False)
    parser.add_argument("-trfr1", "--trajfraction1", help="starting frame fraction of the trajectory (range 0-1) to be used for force calculation (default is 0)", \
                        default=0.0,type=float, required=False)
    parser.add_argument("-trfr2", "--trajfraction2", help="last frame fraction of the trajectory (range 0-1) to be used for force calculation (default is 1)", \
                        default=1.0,type=float, required=False)
    parser.add_argument("-units", "--units", \
                        help="Choose free energy units specifying (case sensitive) either kj (kj/mol) or kcal (kcal/mol) (in alternative you can set the Boltzmann factor through the option -kb)", \
                        type=str, required=False)
    parser.add_argument("-kb", "--kb", help="Boltzmann factor to define free energy units.", \
                        default=-1,type=float, required=False)
    parser.add_argument("-wf", "--widthfactor", help="Scaling factor of the width (wfact) to assign the force constant (k=kb*temp*(wfact*wfact)/(width*width); default is 1 (width is read in the GRID defined in the input file)", \
                        default=1.0,type=float, required=False)
    parser.add_argument("-colv_time_prec", "--colv_time_prec", help="Precision for reading the time column in COLVAR_FILE and HILLS_FILE", \
                        default=-1,type=int, required=False)
    parser.add_argument("-colvarbias_column", "--read_colvarbias_column", help="read biasing force from COLVAR_FILE at a specified number of columns after the associated CV (e.g. would be 1 if it is right after the CV)", \
                        default=-1,type=int, required=False)
    parser.add_argument("-colvars","--colvars", \
                        help="Use default parameters for Colvars", \
                        default=False, dest='do_colv', action='store_true')
    parser.add_argument("-internalf","--internalforces", \
                        help="Provided free energy gradients are based on internal forces", \
                        default=False, dest='do_intern', action='store_true')
    parser.add_argument("-noforce","--nocalcforce", \
                        help="Do not calculate forces. By default forces calculation is ON", \
                        default=True, dest='do_force', action='store_false')
    parser.add_argument("-nopgradb","--noprintgradbias", \
                        help="do not print in output bias gradient trajectory ", \
                        default=True, dest='print_bias', action='store_false')
    parser.add_argument("-obgf", "--outbiasgradfile", \
                        help="output file of bias gradients for each frame", \
                        default="bias_grad.out",type=str, required=False)
    parser.add_argument("-oepf", "--outeffpointsfile", \
                        help="output file of effective points (e.g. on which forces are going to be calculated)", \
                        default="eff_points.out",type=str, required=False)    
    parser.add_argument("-oeff", "--outeffforcefile", \
                        help="output file of free energy gradient on effective points", \
                        default="grad_on_eff_points.out",type=str, required=False)
    parser.add_argument("-oeff1", "--outeffforcefile1", \
                        help="output file of free energy gradient on effective points using first half of the trajectory", \
                        default="grad_on_eff_points1.out",type=str, required=False)
    parser.add_argument("-oeff2", "--outeffforcefile2", \
                        help="output file of of free energy gradient on effective points using second half of the trajectory", \
                        default="grad_on_eff_points2.out",type=str, required=False)
    parser.add_argument("-ocmbeff", "--outcombeffforcefile", \
                        help="output combined file effective points and forces", \
                        default="grad_on_eff_points_comb.out",type=str, required=False)
    parser.add_argument("-olf", "--outlabelfile", \
                        help="output file of labels (assigned bins along colvar) ", \
                        default="label.out",type=str, required=False)
    parser.add_argument("-nobound","--noboundaries", \
                        help="Do not exclude data beyond grid boundaries", \
                        default=True, dest='do_bound', action='store_false')
    parser.add_argument("-nofeffpc","--nofasteffpointcalc", \
                        help="Do not use algorithm for fast calculation of effective points (through binning)", \
                        default=True, dest='do_feffpc', action='store_false')
    parser.add_argument("-noeffpb","--noeffpointbin", \
                        help="Do not calculate effective points by binning but use non-overlapping spherical domains", \
                        default=True, dest='do_effpb', action='store_false')
    parser.add_argument("-jceffp","--justcalceffpoints", \
                        help="Calculate effective points and do nothing else. COLVARS and GRID data must at least be provided", \
                        default=False, dest='do_jceffp', action='store_true')
    parser.add_argument("-jcmetab","--justcalcmetabias", \
                        help="Calculate metadynamics bias potential and do nothing else. COLVARS, HILLS and GRID data must at least be provided", \
                        default=False, dest='do_jmetab', action='store_true')
    parser.add_argument("-label","--label", \
                        help="label COLVARS according to the effective foints", \
                        default=False, dest='do_label', action='store_true')
    parser.add_argument("-wrtlabelscrd","--writelabelscoord", \
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
do_colvars=args.do_colv
do_internalf=args.do_intern
do_label=args.do_label
writelabelscoord=args.write_label_coord
do_force=args.do_force
do_boundaries=args.do_bound
do_gefilter=args.do_gefilt
do_just_eff_points=args.do_jceffp
do_just_hills_bias=args.do_jmetab
do_fast_eff_p_calc=args.do_feffpc
do_bin_eff_p_calc=args.do_effpb
do_backrestart=args.do_backres
print_bias=args.print_bias
bias_grad_file=args.outbiasgradfile
labelfile=args.outlabelfile
eff_points_file=args.outeffpointsfile
force_points_file=args.outeffforcefile
force_points_file_comb=args.outcombeffforcefile
force_points_file1=args.outeffforcefile1
force_points_file2=args.outeffforcefile2
temp=args.temp
skip=args.skip
kb=args.kb
units=args.units
widthfact=args.widthfactor
colv_prec=args.colv_time_prec
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

do_hills_bias=False
do_umbrella_bias=False

if colvarbias_column>0:
  if do_colvars==False:
    print ("ERROR: reading forces from COLVAR files is supported only for colvars ")
    print ("ERROR: if you are using colvars please insert the option -colvars ")
    sys.exit()

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
cvdict={}
ngfiles=0
nefiles=0
nffiles=0
nfcomp=0
read_gfile=False
read_efile=False
read_ffile=False
has_hills=False
gammaf=1
do_us=False
nactive=0
cfile='none'
hfile='none'

# INPUT PARSER
# It reads an input file with relevant information (see manual):
# grid parameters and files with trajectories of collective variables and/or forces, plus other specifications.
# It is structured to be able to read an unlimited number of files. Collective variables specified on the grid can be labeled and recognized
# as specific components on force/gradient trajectory files using a dictionary.  

f=open (ifile, 'r') 
for line in f:
    parts=line.split()
    nparts=len(parts)
    has_c=False
    ncfiles=0
    nhfiles=0
    nafields=0 
    nbffields=0
    nuscvfields=0
    nuskfields=0
    nuscfields=0
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

        has_us=False
        for i in range (0,nparts):
           if str(parts[i])=="US_CVS-CLS":
             lus=i
             has_us=True     
             nuscvfields=nuscvfields+1
        if nuscvfields>1:
          print ("ERROR, US_CVS-CLS must be specified just one time on a single line")
          sys.exit()
        if has_us:
          do_umbrella_bias=True
          nact=0
          do_us=[True] 
          for i in range (lus+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_K" or str(parts[i])=="US_C":
               break
             nact=nact+1
          nactive=[nact]
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0
          for i in range(lus+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_K" or str(parts[i])=="US_C":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1 
          a_cvs=[pippo]

          has_us_k=False
          for i in range (0,nparts):
             if str(parts[i])=="US_K":
               lusk=i
               has_us_k=True
               nuskfields=nuskfields+1
          if has_us_k==False or nuskfields>1:
            print ("ERROR, US_K not specified or specified more than one time on a single line")
            sys.exit()
          if has_us_k:
            pippo=np.zeros((nactive[ncolvars]))
            npippo=0
            for i in range(lusk+1,nparts):
               if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_CVS-CLS" or str(parts[i])=="US_C":
                 break
               pippo[npippo]=float(parts[i])
               npippo=npippo+1
            k_cvs=[pippo]

          has_us_c=False
          for i in range (0,nparts):
             if str(parts[i])=="US_C":
               lusc=i
               has_us_c=True
               nuscfields=nuscfields+1
          if has_us_c==False or nuscfields>1:
            print ("ERROR, US_C not specified or specified more than one time on a single line")
            sys.exit()
          if has_us_c:
            pippo=np.zeros((nactive[ncolvars]))
            npippo=0
            for i in range(lusc+1,nparts):
               if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_CVS-CLS" or str(parts[i])=="US_K":
                 break
               pippo[npippo]=float(parts[i])
               npippo=npippo+1
            c_cvs=[pippo]

        else:
          do_us=[False]


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
          do_hills_bias=True 
          if has_us:
            print ("ERROR, reading HILLS_FILE is not compatible with US")
            sys.exit()
          has_hills=[True]
          hfile=[str(parts[lh+1])]  
          has_a=False
          nact=0 
          has_bf=False
          for i in range (0,nparts):
             if str(parts[i])=="BF":
               lbf=i
               nbffields=nbffields+1
               has_bf=True
          if has_bf==True and nbffields>1:
            print ("ERROR, BF (bias factor) specified more than once on a single line")
            sys.exit()
          if has_bf:
            bias_factor=float(parts[lbf+1])
            if bias_factor<=0:
              print ("ERROR, please select a positive (and non zero) bias factor")
              sys.exit()
            gammaf=[(bias_factor-1.0)/bias_factor]  
          else:
            gammaf=[1.0]
          for i in range (0,nparts):
             if str(parts[i])=="HILLS_CVS-CLS": 
               la=i
               has_a=True
               nafields=nafields+1
          if has_a==False or nafields>1:
            print ("ERROR, HILLS_CVS-CLS not specified or specified more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE" or str(parts[i])=="BF":
               break  
             nact=nact+1
          nactive=[nact]
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0
          for i in range(la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE" or str(parts[i])=="BF":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1 
          a_cvs=[pippo]
        else:
          has_hills=[False]
          hfile=['none']
        if has_h==False and has_us==False:
          nactive=[int(0)] 
          a_cvs=[int(-2)]

      if ncolvars>0:
        cfile.append(str(parts[lc+1])) 

        has_us=False
        for i in range (0,nparts):
           if str(parts[i])=="US_CVS-CLS":
             lus=i
             has_us=True  
             nuscvfields=nuscvfields+1
        if nuscvfields>1:
          print ("ERROR, US_CVS-CLS must be specified just one time on a single line")
          sys.exit()
        if has_us:
          do_umbrella_bias=True
          nact=0  
          do_us.append(True) 
          for i in range (lus+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_K" or str(parts[i])=="US_C":
               break
             nact=nact+1
          nactive.append(nact) 
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0  
          for i in range(lus+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_K" or str(parts[i])=="US_C":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1
          a_cvs.append(pippo)

          has_us_k=False
          for i in range (0,nparts):
             if str(parts[i])=="US_K":
               lusk=i
               has_us_k=True
               nuskfields=nuskfields+1
          if has_us_k==False or nuskfields>1:
            print ("ERROR, US_K not specified or specified more than one time on a single line")
            sys.exit()    
          if has_us_k: 
            pippo=np.zeros((nactive[ncolvars]))
            npippo=0 
            for i in range(lusk+1,nparts):    
               if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_CVS-CLS" or str(parts[i])=="US_C":
                 break 
               pippo[npippo]=float(parts[i])
               npippo=npippo+1    
            k_cvs.append(pippo)
            
          has_us_c=False
          for i in range (0,nparts):
             if str(parts[i])=="US_C":
               lusc=i
               has_us_c=True
               nuscfields=nuscfields+1
          if has_us_c==False or nuscfields>1:
            print ("ERROR, US_C not specified or specified more than one time on a single line")
            sys.exit()
          if has_us_c:    
            pippo=np.zeros((nactive[ncolvars]))
            npippo=0
            for i in range(lusc+1,nparts):
               if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="US_CVS-CLS" or str(parts[i])=="US_K":
                 break
               pippo[npippo]=float(parts[i])
               npippo=npippo+1
            c_cvs.append(pippo)
            
        else:
          do_us.append(False)

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
          do_hills_bias=True 
          if has_us:
            print ("ERROR, reading HILLS_FILE is not compatible with US")
            sys.exit()
          has_hills.append(True)
          hfile.append(str(parts[lh+1]))
          has_a=False
          nact=0   
          has_bf=False
          for i in range (0,nparts):
             if str(parts[i])=="BF":
               lbf=i
               nbffields=nbffields+1
               has_bf=True
          if has_bf==True and nbffields>1:
            print ("ERROR, BF (bias factor) specified more than once on a single line")
            sys.exit()
          if has_bf:
            bias_factor=float(parts[lbf+1])
            if bias_factor<=0:
              print ("ERROR, please select a positive (and non zero) bias factor")
              sys.exit()
            gammaf.append((bias_factor-1.0)/bias_factor)
          else:
            gammaf.append(1.0)
          for i in range (0,nparts):
             if str(parts[i])=="HILLS_CVS-CLS": 
               la=i
               has_a=True
               nafields=nafields+1
          if has_a==False or nafields>1:
            print ("ERROR, HILLS_CVS-CLS not specified or specidied more than once on a single line")
            sys.exit()
          for i in range (la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE" or str(parts[i])=="BF":
               break 
             nact=nact+1
          nactive.append(nact)
          pippo=np.zeros((nactive[ncolvars]),dtype=np.int64)
          npippo=0
          for i in range(la+1,nparts):
             if str(parts[i])=="COLVAR_FILE" or str(parts[i])=="HILLS_FILE" or str(parts[i])=="BF":
               break
             pippo[npippo]=int(parts[i])-1
             npippo=npippo+1
          a_cvs.append(pippo)
        else:
          has_hills.append(False)
          hfile.append('none')
        if has_h==False and has_us==False: 
          nactive.append(int(0))
          a_cvs.append(int(-2))
      ncolvars=ncolvars+1

    if str(parts[0])=="CV-CL":
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
            if len(parts)>6:
              if str(parts[6])=="LABEL":
                if str(parts[7]) in cvdict: 
                  print ("ERROR: CV LABEL",str(parts[7]),"was already used for another CV")
                  sys.exit() 
                cvdict[str(parts[7])]=ndim   
              else:
                print ("Unrecognized character:",str(parts[5]))
                sys.exit()
          else:
            periodic=[0]  
            if str(parts[5])=="LABEL":
              if str(parts[6]) in cvdict:
                print ("ERROR: CV LABEL",str(parts[6]),"was already used for another CV")  
                sys.exit() 
              cvdict[str(parts[6])]=ndim 
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
            if len(parts)>6:
              if str(parts[6])=="LABEL":
                if str(parts[7]) in cvdict:
                  print ("ERROR: CV LABEL",str(parts[7]),"was already used for another CV") 
                  sys.exit()                  
                cvdict[str(parts[7])]=ndim
              else:
                print ("Unrecognized character:",str(parts[5]))
                sys.exit()
          else:
            periodic.append(0)  
            if str(parts[5])=="LABEL":
              if str(parts[6]) in cvdict:
                print ("ERROR: CV LABEL",str(parts[6]),"was already used for another CV")
                sys.exit()                
              cvdict[str(parts[6])]=ndim
            else:
              print ("Unrecognized character:",str(parts[5]))
              sys.exit()
        else:
          periodic.append(0)
      ndim=ndim+1       
    if str(parts[0])=="READ_BIAS_GRAD_TRJ" or str(parts[0])=="READ_BIAS_FORCE_TRJ":
      if colvarbias_column>0:
        print ("ERROR: you are reading the biasing forces two times; from a file (through READ_BIAS_GRAD_TRJ)") 
        print ("and also from the COLVAR_FILE (through -colvarbias_column). Select just the pertinent option. ")
        sys.exit()  
      if ngfiles==0:
        if str(parts[0])=="READ_BIAS_FORCE_TRJ":
          use_force=[-1.0]
        else:
          use_force=[1.0] 
        gfile=[str(parts[1])] 
        read_gfile=True
        do_hills_bias=False
        do_umbrella_bias=False
      if ngfiles>0:
        if str(parts[0])=="READ_BIAS_FORCE_TRJ":
          use_force.append(-1.0)
        else:
          use_force.append(1.0)
        gfile.append(str(parts[1]))     
      if nparts>2:
        if str(parts[2])=="CV_COMP":
          if nparts>3:  
            pluto=np.zeros((nparts-3),dtype=np.int8) 
            if ndim==0:
              print ("ERROR: to read specific components of the biasing force (through CV_COMP) GRID information must be provided before")
              sys.exit()
          else:
            pluto="ALL"
            print ("NO SPECIFIC CV components specified though CV_COMP: assuming the trajectory FORCE/GRAD file includes all components")  
          for i in range (3,nparts):
             if str(parts[i]) in cvdict:
               pluto[i-3]=cvdict[str(parts[i])]
             else:
               tmpwhichcv=np.array(whichcv)
               tmppluto=np.where(tmpwhichcv==int(parts[i])-1)
               pluto[i-3]=np.int8(tmppluto[0])
      else:
        print ("NO SPECIFIC CV components specified though CV_COMP: assuming the trajectory FORCE/GRAD file includes all components")  
        pluto="ALL"  
      if ngfiles==0:
        usecomp=[pluto]
      if ngfiles>0:
        usecomp.append(pluto)
      ngfiles=ngfiles+1
    if str(parts[0])=="READ_EPOINTS":
      if nefiles==0:
        efile=[str(parts[1])]
        read_efile=True
        calc_epoints=False
      if nefiles>0:
        efile.append(str(parts[1])) 
      nefiles=nefiles+1  
    if str(parts[0])=="READ_GRAD_PMF":
      if nffiles==0:
        ffile=[str(parts[1])]
        read_ffile=True        
        read_efile=False
        do_hills_bias=False 
        do_umbrella_bias=False
        calc_force_eff=False
        calc_epoints=False 
      if nffiles>0:
        ffile.append(str(parts[1]))
      pluto=np.zeros((ndim),dtype=np.int8)
      if nparts>2: 
        if str(parts[2])=="REMOVE_COMP":
          if len(pluto)==0:
            print ("ERROR: to use REMOVE_COMP GRID information must be provided before reading the free energy gradients in the input file")
            sys.exit()
          for i in range (3,nparts):
             pluto[np.int8(parts[i])-1]=1
      if nffiles==0:
        rcvcomp=[pluto]
      if nffiles>0:
        rcvcomp.append(pluto)  
      nffiles=nffiles+1

print ("Input read")

if str(units)=="kj":
  kb=0.00831446261815324
elif str(units)=="kcal":
  kb=0.0019858775
elif kb<0 and do_just_eff_points==False and do_just_hills_bias==False and read_ffile==False and do_force==True:
    print ("ERROR: please specify either the units (-units) or the value of the Boltzmann factor (-kb option)")
    sys.exit()

# Below it checks which options have been selected to proceed with different calculations

if colvarbias_column>0:
  read_gfile=True
  do_hills_bias=False
  do_umbrella_bias=False
  if do_gefilter:
    print ("ERROR: filtering according to colvar energy match is not compatible with reading applied forces from colvar file (as those forces are expected to be exact)") 
    sys.exit() 

if do_just_eff_points:
  calc_epoints=True
  do_hills_bias=False
  do_umbrella_bias=False
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False  
  print ("Requested just derivation of effective points (e.g. GRID on which mean forces will be evaluated) and nothing else")

if do_just_hills_bias:
  calc_epoints=False
  do_hills_bias=True
  do_umbrella_bias=False
  read_gfile=False
  read_efile=False
  read_ffile=False
  calc_force_eff=False
  print ("Requested just derivation applied forces from HILLS files for each simulation frame and nothing else")
  
if ncolvars==0:
  calc_epoints=False  
  calc_force_eff=False

if do_force==False:
  calc_force_eff=False  

if do_hills_bias:
  do_internalf=False
  do_umbrella_bias=False

if do_umbrella_bias:
  do_internalf=False
  do_hills_bias=False


internalf=1.0
if do_internalf:
  internalf=0.0
  print ("The force for each frame read from file (through READ_BIAS_GRAD_TRJ or -colvarbias_column) is the ") 
  print ("internal force: FCAM not applied, mean forces are calculated as the local average of the internal forces") 
  if colvarbias_column<=1:
    print ("ERROR: both total and applied force must be provided to evaluate the internal force; -colvarbias_column must be larger than 1")
    sys.exit()    

if read_gfile==False:
  if do_hills_bias==False and do_umbrella_bias==False:
    if calc_force_eff: 
      print ("NOTE: no option for reading applied forces from files or calculating them (through HILLS files") 
      print ("      or harmonic umbrella) was selected: mean forces will be calculated assuming UNBIASED SAMPLING.")

if ndim==0:
  print ("ERROR: number of variables is zero, please provide some to continue")
  sys.exit()
       
if calc_force_eff and temp<0:
  print ("ERROR: temperature for calculating forces not provided or negative value")
  sys.exit()
 
if read_gfile:
  if colvarbias_column<=0:
    for k in range (0,ngfiles):
       if str(usecomp[k])!="ALL": 
         if do_gefilter:
           print ("ERROR: gaussian energy filter is not supported when FORCE/GRAD components to read are specified")
           sys.exit()

    if ngfiles!=1:
        try:
            assert(ngfiles == ncolvars)
        except AssertionError:
            raise FCAM_Error("Please provide a unique gradient file or a gradient file for each colvar. Note that in either case this must be consistent with the ORDERED set of colvar files provided")
                
if do_hills_bias or do_umbrella_bias:
  if print_bias:
    with open(bias_grad_file, 'w') as f:
        f.write("# Time, grad, Gaussenergy, numhill, next_restart, previous_restart, replica \n")

if calc_epoints:
  with open(eff_points_file, 'w') as f:
      f.write("# numeff, colvar, npoints \n")

if do_label:
  if writelabelscoord==False:
    with open(labelfile, 'w') as f:
        f.write("# time, label, ntraj \n") 
  else:
    with open(labelfile, 'w') as f:
        f.write("# time, colvar, label, ntraj \n")
     
has_hills=np.array(has_hills)
do_us=np.array(do_us)
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

try:
    assert(len(cunique)==len(cfile))
except AssertionError:
    try:
        assert(cunique[0]=="none")
    except AssertionError:    
        raise FCAM_Error("Same COLVAR file introduced multiple times in the input")

try:
    assert(len(hunique)==len(hfile))
except AssertionError:
    try:
        assert(hunique[0]=="none")
    except AssertionError:
        raise FCAM_Error("Same HILLS file introduced multiple times in the input")

try:
    assert(len(vunique)==len(whichcv))
except AssertionError:
    raise FCAM_Error("Same CV introduced multiple times in the input")

#if len(vunique)!=len(whichcv):
#  print ("ERROR: same CV introduced multiple times in the input")
#  sys.exit()

iactive=np.zeros((ncolvars,ndim),dtype=np.int8)
iactive[:,:]=-1

allfound=True
for i in range (0,ncolvars):
   if nactive[i]>ndim:
     print ("ERROR: number of HILLS_CVS-CLS or US_CVS-CLS larger than total number of CVS")
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
  print ("ERROR: HILLS_CVS-CLS or US_CVS-CLS must be part of the CVS used for CV-CL")
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

# Below there are a set of routines to calculate the number of effective points on which forces will be calculated.
# The fastest and default routune is fast_calc_eff_points, which provides points on a regular grid.
# Note that effective points that are not explored (far from sampled ones) are excluded.
# TO DO: might be useful to do fast a routine for a non regular grid, e.g. using Daura or similar clustering scheme, that 
# better captures the space topology.

# This routine calculates effective points based on spherical cut-off (upperbound-lowerbound/number of points in each direction).
# Doesn't give memory issues or overflow but it's slow in high dimensionality.
@jit(nopython=True)
def calc_eff_points(numpoints, inputarray, npointsins):
   diffc=np.zeros((numpoints,ndim))
   distance=np.zeros((numpoints))
   numinbin=np.zeros((numpoints),dtype=np.int64)
   neffp=1 
   ceff=np.zeros((numpoints,ndim))
   ceff[0,:]=inputarray[0,:]
   numinbin[0]=npointsins[0]
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

# This routine calculates effective points on a regular grid. Does not give memory issues or overflow
# but is slow in high dimensionality. 
@jit(nopython=True)
def calc_eff_points_bin(numepoints, effparray, npointsins):
   colvarsbineff=np.zeros((numepoints, ndim))
   nbins=1
   mywidth=width
   diffc=np.zeros((numepoints,ndim))
   distance=np.zeros((numepoints))
   diffbin=np.zeros(ndim)
   colvarbin=np.zeros(ndim)
   numinbin=np.zeros((numepoints),dtype=np.int64)
   totperiodic=np.sum(periodic[0:ndim])
   diffc[0,:]=effparray[0,:]-lowbound
   bingrid=np.floor(diffc[0,:]/mywidth)
   if totperiodic>0:
     tmpdiffc=mywidth*(0.5+bingrid)-0.5*box
     tmpdiffc=tmpdiffc/box[0:ndim]
     tmpdiffc=tmpdiffc-np.rint(tmpdiffc)*periodic[0:ndim]
     tmpdiffc=tmpdiffc*box[0:ndim]
     bingrid=np.rint(((tmpdiffc+0.5*box)/mywidth)-0.5)
   colvarbin=lowbound+mywidth*(0.5+bingrid)
   colvarsbineff[0,:]=colvarbin
   numinbin[0]=npointsins[0]
   for i in range(1,numepoints):
      diffc[i,:]=effparray[i,:]-lowbound
      bingrid=np.floor(diffc[i,:]/mywidth)
      if totperiodic>0:
        tmpdiffc=mywidth*(0.5+bingrid)-0.5*box
        tmpdiffc=tmpdiffc/box[0:ndim]
        tmpdiffc=tmpdiffc-np.rint(tmpdiffc)*periodic[0:ndim]
        tmpdiffc=tmpdiffc*box[0:ndim]
        bingrid=np.rint(((tmpdiffc+0.5*box)/mywidth)-0.5)
      colvarbin=lowbound+mywidth*(0.5+bingrid)     
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

# This routine calculates effective points on a regular grid. It is based on setting a bin identity number given by 
# the sum of the bin grid vectors (integer associated to each bin in each component) scaled by specific factors 
# for each component. In this manner, effective bins are associated unique identity numbers.
# The routine is very fast and can be used up to large dimensionality (around 10). Might give overflow at dimensionality beyond 10.
# It is fully tested and reliable.
# 

def fast_calc_eff_points(numepoints, effparray, npointsins):
   diffc=np.zeros((numepoints,ndim))
   myshift=np.ones((ndim),dtype=np.int64)
   mywidth=width

   newnpointsv=npointsv+1
   newlowbound=lowbound
   newupbound=upbound
   if do_boundaries==False:
     newlowbound=np.amin(effparray, axis=0)
     newupbound=np.amax(effparray, axis=0) 
     newlowbound=np.where(newlowbound<lowbound,newlowbound,lowbound)
     newupbound=np.where(newupbound>upbound,newupbound,upbound)
     newnpointsv=1+np.rint((newupbound-newlowbound)/mywidth)

   shift=1
   myshift[0]=1
   binid=np.ones((numepoints),dtype=np.int64)
   for i in range (1,ndim):
      myshift[i]=shift*newnpointsv[i-1]
      shift=myshift[i]
   diffc[:,:]=effparray[:,:]-lowbound
   bingrid=np.floor(diffc[:,:]/mywidth)
   # if periodic check that bin is not out of periodic boundaries (e.g. when value is exactly upper boundary)
   # otherwise put it back into the periodic interval
   totperiodic=np.sum(periodic[0:ndim]) 
   if totperiodic>0:
     diffc[:,:]=mywidth*(0.5+bingrid)-0.5*box
     diffc[:,:]=diffc[:,:]/box[0:ndim]
     diffc[:,:]=diffc[:,:]-np.rint(diffc[:,:])*periodic[0:ndim] 
     diffc[:,:]=diffc[:,:]*box[0:ndim]
     bingrid=np.rint(((diffc[:,:]+0.5*box)/mywidth)-0.5)   
   binid=np.sum(bingrid*myshift,axis=1)
   effbinid=np.unique(binid,return_index=True,return_inverse=True,return_counts=True)
   neffp=effbinid[1].size
   indexbin=np.argsort(effbinid[2][:])
   effcount=np.cumsum(effbinid[3])
   effindexbin=np.split(indexbin,effcount)
   numinpoints=np.zeros((neffp),dtype=np.int64)
   for i in range (0,neffp):
      numinpoints[i]=np.sum(npointsins[effindexbin[i]])
   colvarbin=lowbound+mywidth*(0.5+bingrid[effbinid[1][:],:])

   # to do: mask bins out of perdiodic boundaries (if found)

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


#ROUTINE TO CALCULATE THE FORCE ON A SET OF POINTS BASED ON THE FORCE-CORRECTION ANALYSIS METHOD (FCAM)

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
      weight=np.exp(-distance[whichpoints[0][:]])
      if weight.size>0:
        grade[i,:]=-np.sum((internalf*widthfact*kb*temp*diffc[whichpoints[0][:],:]/width[:]+gradbias[whichpoints[0][:],:])*weight[:,np.newaxis],axis=0) 
        tweights[i]=np.sum(weight)
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

# Rountine to calculate instantaneous biasing forces for metadynamics simulations from HILLS files (files with history of the Gaussians added).
# Note that this will entail binning errors if the metdadynamics code stores forces on a grid.
# The output must be carefully checked if trajectories are not restarted exactly. The code try to find an optimal solution, 
# so in many cases it will work finem but double check is recommended.
# For instance, check that the output Gaussian energy from the code below is the same of the original metadynamics code. 
# Multiple HILLS files can be specified alongside selected biased coordinates (see manual).
 
if do_hills_bias:
  print ("Calculating metadynamics bias forces on each COLVAR point of the selected variables from the HILLS files")
  print ("which store the deposited Gaussian hills (TIME CV1 CV2... SIGMA_CV1 SIGMA_CV2... HEIGHT)")
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
     if nactive[k]>0 and has_hills[k]:
       countinter=0
       numhills=0
       current_hill_time=0.0
       prev_hill_time=-1.0
       current_colv_time=0.0
       prev_colv_time=-1.0 
       for i in range(0,npoints[k]):
          whichhills_old=whichhills
          if colv_prec>0:
            current_colv_time=np.round(colvarsarray[k][i,0],colv_prec)
            if i>0: 
              prev_colv_time=np.round(colvarsarray[k][i-1,0],colv_prec)
            if numhills>0 and numhills<nhills[k]: 
              current_hill_time=np.round(hillsarray[k][numhills,0],colv_prec)
              prev_hill_time=np.round(hillsarray[k][numhills-1,0],colv_prec)
          else:
            current_colv_time=colvarsarray[k][i,0]
            if i>0: 
              prev_colv_time=colvarsarray[k][i-1,0]
            if numhills>0 and numhills<nhills[k]:
              current_hill_time=hillsarray[k][numhills,0]
              prev_hill_time=hillsarray[k][numhills-1,0]
          if i>0 and current_colv_time<prev_colv_time:
            if numhills>0 and numhills<nhills[k] and prev_hill_time>current_hill_time: 
              index=trh
              whichhills_old=np.where(dvec<trh,1,0)
          if do_large_hfreq:
            if numhills>=trh:
              index=trh
              whichhills_old=np.where(dvec<trh,1,0)
          if colv_prec>0:
            trh=1+index+np.array(np.where(np.round(hillsarray[k][index+1:nhills[k],0],colv_prec)-np.round(hillsarray[k][index:nhills[k]-1,0],colv_prec)<0))
          else:
            trh=1+index+np.array(np.where(hillsarray[k][index+1:nhills[k],0]-hillsarray[k][index:nhills[k]-1,0]<0))
          if trh.size!=0:
            if colv_prec>0:
              trh=np.amin(2+index+np.array(np.where(np.round(hillsarray[k][index+2:nhills[k],0],colv_prec)-np.round(hillsarray[k][index+1:nhills[k]-1,0],colv_prec)<0)))
            else:
              trh=np.amin(2+index+np.array(np.where(hillsarray[k][index+2:nhills[k],0]-hillsarray[k][index+1:nhills[k]-1,0]<0)))
          else:
            trh=nhills[k]
          if colv_prec>0:
            whichhills=np.where(np.round(hillsarray[k][:,0],colv_prec)<=np.round(colvarsarray[k][i,0],colv_prec),1,0) 
          else:
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
            expdiff[0:numhills]=gammaf[k]*hillsarray[k][0:numhills,2*nactive[k]+1]*expdiff[0:numhills]
            gaussenergy[k][i]=np.sum(expdiff[0:numhills])
            for j in range(0,nactive[k]):  
               gradv[k][i,iactive[k,j]]=np.sum(diff[0:numhills,j]*expdiff[0:numhills]/hillsarray[k][0:numhills,nactive[k]+1+j],axis=0)
          if print_bias: 
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
       if print_bias: 
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

# This routine calculates the instantaneous biasing forces for umbrella sampling calculations from the provided force constants (see manual).
if do_umbrella_bias:
  print ("Calculating umbrella sampling bias forces on each COLVAR point of the selected variables from the centers and force constants provided")

  for k in range (0,ncolvars):
     diff=np.zeros((nactive[k]))
     if nactive[k]>0 and do_us[k]:
       for i in range(0,npoints[k]):
          diff[0:nactive[k]]=colvarsarray[k][i,a_cvs[k][0:nactive[k]]+1]-c_cvs[k][0:nactive[k]]
          diff[0:nactive[k]]=diff[0:nactive[k]]/box[iactive[k,0:nactive[k]]]
          diff[0:nactive[k]]=diff[0:nactive[k]]-np.rint(diff[0:nactive[k]])*periodic[iactive[k,0:nactive[k]]]
          diff[0:nactive[k]]=diff[0:nactive[k]]*box[iactive[k,0:nactive[k]]]
          gradv[k][i,iactive[k,0:nactive[k]]]=k_cvs[k][0:nactive[k]]*diff[0:nactive[k]]
          if print_bias:
            with open(bias_grad_file, 'a') as f:
                f.write("%s " % (colvarsarray[k][i,0]))
                for j in range(0,ndim):
                   f.write("%s " % (gradv[k][i,j]))
                f.write("%s " % " 0 ")
                f.write("%s " % " 0 ")
                f.write("%s " % " 0 ")
                f.write("%s " % " 0 ")
                f.write("%s \n" % (k))
     else:
       if print_bias:
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
# The program can read multiple files with all components or with specified ones.
# It can also read a single file with appended trajectories. In either case the order must be the same of the COLVAR files.       
if read_gfile:
  print ("Reading applied forces per frame from external file...") 
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
           if str(usecomp[0])=="ALL":       
             gradv=[use_force[0]*gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1]]
             if do_gefilter:
               gaussenergy=[gradarray[0][totpoints:totpoints+npoints[k],ndim+1]]
           else:
             tmpgradv=np.zeros((npoints[k],ndim))
             for j in range(0,len(usecomp[0])):
                tmpgradv[:,usecomp[0][j]]=use_force[0]*gradarray[0][totpoints:totpoints+npoints[k],j+1]
             gradv=[tmpgradv]            
         if k>0:
           if str(usecomp[0])=="ALL":  
             gradv.append(use_force[0]*gradarray[0][totpoints:totpoints+npoints[k],1:ndim+1])
             if do_gefilter:
               gaussenergy.append(gradarray[0][totpoints:totpoints+npoints[k],ndim+1]) 
           else:
             tmpgradv=np.zeros((npoints[k],ndim))
             for j in range(0,len(usecomp[0])):
                tmpgradv[:,usecomp[0][j]]=use_force[0]*gradarray[0][totpoints:totpoints+npoints[k],j+1]
             gradv.append(tmpgradv)
         totpoints=totpoints+npoints[k]
    else:
      print ("Reading separate external files with applied force for each COLVAR file ...") 
      for k in range (0,ngfiles):
         if k==0:
           gradarray = [np.loadtxt(gfile[k])]
           if npoints[k]!=len(gradarray[k]):
             print ("ERROR: gradient file doesn't match COLVAR file")
             sys.exit()
           if str(usecomp[k])=="ALL":  
             gradv=[use_force[k]*gradarray[k][:,1:ndim+1]]
             if do_gefilter:
               gaussenergy=[gradarray[k][:,ndim+1]]
           else:
             tmpgradv=np.zeros((npoints[k],ndim))
             for j in range(0,len(usecomp[0])):
                tmpgradv[:,usecomp[k][j]]=use_force[k]*gradarray[k][:,j+1] 
             gradv=[tmpgradv]
         if k>0:
           gradarray.append(np.loadtxt(gfile[k]))
           if npoints[k]!=len(gradarray[k]):
             print ("ERROR: gradient file doesn't match COLVAR file")
             sys.exit()
           if str(usecomp[k])=="ALL":
             gradv.append(use_force[k]*gradarray[k][:,1:ndim+1])
             if do_gefilter:
               gaussenergy.append(gradarray[k][:,ndim+1])
           else:
             tmpgradv=np.zeros((npoints[k],ndim))
             for j in range(0,len(usecomp[0])):
                tmpgradv[:,usecomp[k][j]]=use_force[k]*gradarray[k][:,j+1]
             gradv.append(tmpgradv)
             
  if colvarbias_column>0:
    tmpwhichcv=np.array(whichcv)
    for k in range (0,ncolvars):
       if k==0:
         gradv=[-colvarsarray[k][:,tmpwhichcv[0:ndim]+1+colvarbias_column]]
       if k>0:
         gradv.append(-colvarsarray[k][:,tmpwhichcv[0:ndim]+1+colvarbias_column])
          
# Some filtering if requested:create masked colvarsarray and gradv eliminating frames before and after restart

if nfrestart>0:
  for i in range (0,ncolvars):
     getpoints=np.where(colvarsarray[i][1:npoints[i],0]<=colvarsarray[i][0:npoints[i]-1,0])
     getipoints=getpoints[0]-nfrestart+1
     if do_backrestart:
       print ("Removing ",nfrestart," frames before trajectory restart for traj ",i)
       getfpoints=getpoints[0]+1
     else:
       print ("Removing ",nfrestart," frames before and ", nfrestart," frames after trajectory restart for traj ",i)
       getfpoints=getpoints[0]+nfrestart+1
     for j in range (0,getpoints[0].size): 
        if getipoints[j]<0:
          getipoints[j]=getpoints[0][j]      
        colvarsarray[i][getipoints[j]:getfpoints[j],0]=np.nan
  #Debug
  #for i in range (0,npoints[i]):
  #   print colvarsarray[0][i,0],colvarsarray[0][i,1],colvarsarray[0][i,2],gradv[0][i,0],gradv[0][i,1] 
  #sys.exit()     

# More filtering if requested: create masked colvarsarray and gradv eliminating values beyond boundaries
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
  #   print (colvarsarray[0][i,0],colvarsarray[0][i,1],colvarsarray[0][i,2],gradv[0][i,0],gradv[0][i,1]) 
  #sys.exit()     


# Other filtering: create masked colvarsarray and gradv filtering values according to the difference between the calculated Gaussian energy
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
   if np.ma.is_masked(colvarsarray[i]):
     print ("NUMBER OF POINTS FOR WALKER ",i,": ",len(colvarsarray[i][:,0][~colvarsarray[i][:,0].mask]))
   else:
     print ("NUMBER OF POINTS FOR WALKER ",i,": ",len(colvarsarray[i][:,0]))

# CALCULATE NUMBER OF EFFECTIVE POINTS IF REQUESTED
if calc_epoints:
  print ("Calculating effective points...")
  for k in range (0,ncolvars):
     if np.ma.is_masked(colvarsarray[k]):
       ntotpoints=len(colvarsarray[k][:,0][~colvarsarray[k][:,0].mask])
     else:
       ntotpoints=len(colvarsarray[k][:,0])
     if k==0:    
       colvarseff=[np.zeros((ntotpoints,ndim))]
       numinpoint=[np.ones((ntotpoints),dtype=np.int64)] 
       neffpoints=[1]
     else:
       colvarseff.append(np.zeros((ntotpoints,ndim)))
       numinpoint.append(np.ones((ntotpoints),dtype=np.int64))
       neffpoints.append(1)      
     arrayin=np.zeros((ntotpoints,ndim))
     if np.ma.is_masked(colvarsarray[k]):
       for j in range (0,ndim):
          arrayin[:,j]=colvarsarray[k][:,whichcv[j]+1][~colvarsarray[k][:,whichcv[j]+1].mask]
     else:
       for j in range (0,ndim):
          arrayin[:,j]=colvarsarray[k][:,whichcv[j]+1] 
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

# READ EFFECTIVE POINTS FROM FILE: multiple files can be specified and the code will merge them. 
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
         print("ERROR: not valid effective file (tip: check that input file name is correct)")
         sys.exit()

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

# CALC FORCE ON EFFECTIVE POINTS USING FCAM

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
     if np.ma.is_masked(colvarsarray[k]) and np.ma.is_masked(gradv[k]): 
       ntotpoints=len(colvarsarray[k][initframe:lastframe:skip,0][~colvarsarray[k][initframe:lastframe:skip,0].mask])
       ntotpoints_grad=len(gradv[k][initframe:lastframe:skip,0][~gradv[k][initframe:lastframe:skip,0].mask])
     else:
       ntotpoints=len(colvarsarray[k][initframe:lastframe:skip,0])
       ntotpoints_grad=len(gradv[k][initframe:lastframe:skip,0])
     if ntotpoints!=ntotpoints_grad:
       print ("ERROR: colvar file and gradient file do not match")  
     arrayin=np.zeros((ntotpoints,ndim))
     gradvin=np.zeros((ntotpoints,ndim))
     if np.ma.is_masked(colvarsarray[k]) and np.ma.is_masked(gradv[k]):
       for j in range (0,ndim):
          arrayin[:,j]=colvarsarray[k][initframe:lastframe:skip,whichcv[j]+1][~colvarsarray[k][initframe:lastframe:skip,whichcv[j]+1].mask]
          gradvin[:,j]=gradv[k][initframe:lastframe:skip,j][~gradv[k][initframe:lastframe:skip,j].mask]
     else:
       for j in range (0,ndim):
          arrayin[:,j]=colvarsarray[k][initframe:lastframe:skip,whichcv[j]+1]
          gradvin[:,j]=gradv[k][initframe:lastframe:skip,j]
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
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         f.write("%s \n " % (weighttot[i]))
  for i in range(0,neffpoints):
     with open(force_points_file1, 'a') as f:
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad1[i,j]))
         f.write("%s \n " % (weighttot1[i]))
  for i in range(0,neffpoints):
     with open(force_points_file2, 'a') as f:
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


# READ FORCE AND EFFECTIVE POINTS FROM EXTERNAL FILE.
# Multiple files can be specified and the code will combine them 
# giving a best estimate. This also provides a trivial parallelization scheme
# where forces can be calculated from individual trajectories and then merged together. 
if read_ffile:
  print ("Reading effective points and forces from external file...")
  if nffiles>1:
    print ("Reading forces arising from ",nffiles," simulations.")
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
           # REMOVE BIAS COMPONENTS IF REQUESTED
           if len(rcvcomp[n])>0: 
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
           # REMOVE BIAS COMPONENTS IF REQUESTED
           if len(rcvcomp[n])>0: 
             for j in range(0,ndim):
                if rcvcomp[n][j]==1:
                  weightr[:,j]=np.zeros((len(tryarray)))     
           weighttot=weighttot+weightr 
           gradr=tryarray[:,ndim:2*ndim]
           for j in range(0,ndim):
              grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradr[0:neffpoints,j]*weightr[0:neffpoints,j]
     except IOError:
         print("ERROR: not valid gradient file (tip: check that input file names are correct)")
         sys.exit() 
  for j in range(0,ndim):
     grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints,j]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints,j]),0)
  for i in range(0,neffpoints):
     with open(force_points_file_comb, 'a') as f:
         for j in range (0,ndim):
            f.write("%s " % (colvarseff[i,j]))
         for j in range (0,ndim):
            f.write("%s " % (grad[i,j]))
         for j in range (0,ndim-1):
            f.write("%s " % (weighttot[i,j])) 
         f.write("%s \n " % (weighttot[i,ndim-1]))

print("--- %s seconds ---" % (time.time() - start_time)) 

