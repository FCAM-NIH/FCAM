import numpy as np
import argparse, os, sys
from glob import glob
from copy import deepcopy

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--colvarfile", \
                        help="COLVAR file for analysis", \
                        type=str, required=True)
    parser.add_argument("-h","--hillsfile", \
                        help="HILLS file for metadynamics analysis (units kJ/mol).", \
                        type=str,required=False)
    parser.add_argument("-ndim", "--ndim", help="Number of collective variables. No default", \
                        type=int, required=True)
    parser.add_argument("-temp", "--temp", help="Temperature. No default", \
                        type=int, required=True)


# parameters
hcutoff=6.25 # cutoff for Gaussians
kb=0.00831446261815324


# Variables
ndim = args.ndim
temp = args.temp
hfile = args.hillsfile
cfile = args.colvarfile

#with open("calcf_output1.dat" , 'w') as f:
#    f.write("# Time, grad,Gaussenergy \n")

#with open("grad_eff_output0.dat" , 'w') as f:
#    f.write("# Time, grad,neff-points \n")

#with open("colvar_eff.dat" , 'w') as f:
#    f.write("# numeff, colvar \n")

with open("grad_combined.dat" , 'w') as f:
    f.write("# Time, grad \n")

with open("grad_binned.dat" , 'w') as f:
    f.write("# Time, grad \n")


# read hills file header to find active CVs

count=0
f=open (hfile, 'r')
for line in f:
   count=count+1
   if count==1:    
     parts=line.split()
     nactive=int(parts[1])
     a_cvs=np.zeros((nactive),dtype=np.int32)
     for i in range(0,nactive):
        a_cvs[i]=int(parts[i+2])-1 

iactive=np.zeros((ndim),dtype=np.int8)

iactive[a_cvs[:]]=1

# read the hills file (for now read just one colvars and one hills file)

hillsarray = np.loadtxt(hfile)

nhills=len(hillsarray)

tmp_cvhills=hillsarray[:,1:nactive+1]
tmp_deltahills=hillsarray[:,nactive+1:2*nactive+1]
whills=hillsarray[:,2*nactive+1]

cvhills=np.zeros((nhills,ndim))
deltahills=np.ones((nhills,ndim))

cvhills[:,a_cvs[0:nactive]]=tmp_cvhills[:,0:nactive]
deltahills[:,a_cvs[0:nactive]]=tmp_deltahills[:,0:nactive]

# read the colvar file

colvarsarray = np.loadtxt(cfile)

# now read the header of grid file
header=np.zeros((ndim,4))
count=0
f=open ("grid_data.dat", 'r')
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


npoints=len(colvarsarray)

diff=np.zeros((nhills,ndim))
diff2=np.zeros((nhills,ndim))
gradv=np.zeros((npoints,ndim))

#expdiff=np.zeros((nhills))
#whichhills=np.zeros((nhills),dtype=np.int32)
#for i in range(0,npoints):
#   whichhills[:]=np.where(hillsarray[:,0]<=colvarsarray[i,0],1,0)
#   numhills=np.sum(whichhills) 
#   diff[0:numhills,0:ndim]=(cvhills[0:numhills,0:ndim]-colvarsarray[i,1:ndim+1])*iactive[0:ndim] 
#   diff[0:numhills,0:ndim]=diff[0:numhills,0:ndim]/box[0:ndim]
#   diff[0:numhills,0:ndim]=diff[0:numhills,0:ndim]-np.rint(diff[0:numhills,0:ndim])*periodic[0:ndim]
#   diff[0:numhills,0:ndim]=diff[0:numhills,0:ndim]*box[0:ndim]
#   diff[0:numhills,0:ndim]=diff[0:numhills,0:ndim]/deltahills[0:numhills,0:ndim]
#   diff2[0:numhills,0:ndim]=diff[0:numhills,0:ndim]*diff[0:numhills,0:ndim]
#   diff2[0:numhills,0:ndim]=0.5*diff2[0:numhills,0:ndim]
#   expdiff[0:numhills]=np.sum(diff2[0:numhills,0:ndim]*iactive[0:ndim],axis=1)
#   expdiff[0:numhills]=np.where(expdiff[0:numhills]<hcutoff,np.exp(-expdiff[0:numhills]),0.0) 
#   #expdiff[0:numhills]=np.where(hillsarray[:,0]<=colvarsarray[i,0],expdiff[:],0.0)
#   expdiff[0:numhills]=whills[0:numhills]*expdiff[0:numhills]
#   gaussenergy=np.sum(expdiff[0:numhills])
#   for j in range(0,ndim):
#      gradv[i,j]=iactive[j]*np.sum(diff[0:numhills,j]*expdiff[0:numhills]/deltahills[0:numhills,j])
#   with open("calcf_output1.dat", 'a') as f:
#       f.write("%s %s %s %s \n" % (colvarsarray[i,0],gradv[i,0],gradv[i,1],gaussenergy))


#sys.exit()
# DEBUG

gradarray = np.loadtxt("calcf_output0.dat")
gradv=gradarray[:,1:ndim+1]

# END DEBUG


diffc=np.zeros((npoints,ndim))
grad=np.zeros((npoints,ndim))
#distance=np.zeros((npoints))

# CALCULATE NONBER OF EFFECTIVE POINTS

#colvarseff=np.zeros((npoints,ndim))

#neffpoints=1
#colvarseff[0,:]=colvarsarray[0,1:ndim+1]
#with open("colvar_eff1.dat", 'a') as f:
#    f.write("%s %s %s \n" % (neffpoints,colvarseff[neffpoints-1,0],colvarseff[neffpoints-1,1]))
#for i in range(1,npoints):
#   diffc[0:neffpoints,:]=colvarseff[0:neffpoints,:]-colvarsarray[i,1:ndim+1]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]/box[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]-np.rint(diffc[0:neffpoints,:])*periodic[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]*box[0:ndim]
#   #diffc[0:neffpoints,:]=2.0*diffc[0:neffpoints,:]/deltahills[0,0:ndim]
#   diffc[0:neffpoints,:]=2.0*diffc[0:neffpoints,:]/width[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]*diffc[0:neffpoints,:]
#   distance[0:neffpoints]=np.sum(diffc[0:neffpoints,:],axis=1) 
#   mindistance=np.amin(distance[0:neffpoints])
#   if mindistance>1:
#     neffpoints=neffpoints+1
#     colvarseff[neffpoints-1,:]=colvarsarray[i,1:ndim+1] 
#     with open("colvar_eff1.dat", 'a') as f:
#         f.write("%s %s %s \n" % (neffpoints,colvarseff[neffpoints-1,0],colvarseff[neffpoints-1,1]))

# concatenate multiple effective colvars and recalculate effective points

nrep=2

#colvareffarray = np.loadtxt("colvar_eff0.dat")

#for n in range(1,nrep):
#    try:
#        tryarray = np.loadtxt("colvar_eff%s.dat" % n)
#        colvareffarray=np.concatenate((colvareffarray,tryarray),axis=0)
#    except IOError:
#        pass

#print "effective colvars read"

# recalculate effective points

#nepoints=len(colvareffarray)

#diffc=np.zeros((nepoints,ndim))
#grad=np.zeros((nepoints,ndim))
#distance=np.zeros((nepoints))

#colvarseff=np.zeros((nepoints,ndim))

#neffpoints=1
#colvarseff[0,:]=colvareffarray[0,1:ndim+1]
#with open("colvar_eff.dat", 'a') as f:
#    f.write("%s %s %s \n" % (neffpoints,colvarseff[neffpoints-1,0],colvarseff[neffpoints-1,1]))
#for i in range(1,nepoints):
#   diffc[0:neffpoints,:]=colvarseff[0:neffpoints,:]-colvareffarray[i,1:ndim+1]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]/box[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]-np.rint(diffc[0:neffpoints,:])*periodic[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]*box[0:ndim]
#   #diffc[0:neffpoints,:]=2.0*diffc[0:neffpoints,:]/deltahills[0,0:ndim]
#   diffc[0:neffpoints,:]=2.0*diffc[0:neffpoints,:]/width[0:ndim]
#   diffc[0:neffpoints,:]=diffc[0:neffpoints,:]*diffc[0:neffpoints,:]
#   distance[0:neffpoints]=np.sum(diffc[0:neffpoints,:],axis=1)
#   mindistance=np.amin(distance[0:neffpoints])
#   if mindistance>1:
#     neffpoints=neffpoints+1
#     colvarseff[neffpoints-1,:]=colvareffarray[i,1:ndim+1]
#     with open("colvar_eff.dat", 'a') as f:
#         f.write("%s %s %s \n" % (neffpoints,colvarseff[neffpoints-1,0],colvarseff[neffpoints-1,1]))


#sys.exit()

# CALC FORCE ON EFFECTIVE POINTS

# DEBUG read generated effective points

effarray = np.loadtxt("colvar_eff.dat")
neffpoints=len(effarray)
colvarseff=np.zeros((neffpoints,ndim))
colvarseff[0:neffpoints,0:ndim]=effarray[0:neffpoints,1:ndim+1]
grad=np.zeros((neffpoints,ndim))
# END DEBUG

#for i in range(0,neffpoints):
#   diffc=colvarsarray[0:npoints,1:ndim+1]-colvarseff[i,0:ndim]
#   diffc=diffc/box[0:ndim] 
#   diffc=diffc-np.rint(diffc)*periodic[0:ndim]
#   diffc=diffc*box[0:ndim]
#   #diffc=2.0*diffc/deltahills[0,0:ndim]
#   diffc=2.0*diffc/width[0:ndim]
#   for j in range(0,ndim):
#      weight=np.exp(np.sum(-0.5*diffc[0:npoints,:]*diffc[0:npoints,:],axis=1)) 
#      grad[i,j]=-np.average(2.0*kb*temp*diffc[0:npoints,j]/width[j]+gradv[0:npoints,j],weights=weight)
#      #grad[i,j]=-np.average(2.0*kb*temp*diffc[0:npoints,j]/deltahills[0,j]+gradv[0:npoints,j],weights=np.exp(np.sum(-0.5*diffc[0:npoints,:]*diffc[0:npoints,:],axis=1)))
#   with open("grad_eff_output0.dat", 'a') as f:
#       f.write("%s %s %s %s %s %s \n" % (i,colvarseff[i,0],colvarseff[i,1],grad[i,0],grad[i,1],np.sum(weight)))

# Combine different replicas

gradrep=np.zeros((nrep,neffpoints,ndim))
weightrep=np.zeros((nrep,neffpoints))
grad=np.zeros((neffpoints,ndim))
for n in range(0,nrep):
   tryarray = np.loadtxt("grad_eff_output%s.dat" % n)
   gradrep[n,:,:]=tryarray[:,ndim+1:2*ndim+1]
   weightrep[n,:]=tryarray[:,2*ndim+1]

   #weightrep[n,:]=np.where(weightrep[n,:]<=1,0,weightrep[n,:])
   for j in range(0,ndim):
      grad[0:neffpoints,j]=grad[0:neffpoints,j]+gradrep[n,0:neffpoints,j]*weightrep[n,0:neffpoints]

#grad=grad/(np.sum(weightrep[:,0:neffpoints,np.newaxis],axis=0))

#for j in range(0,ndim):
#   grad[0:neffpoints,j]=np.average(gradrep[:,0:neffpoints,j],axis=0,weights=weightrep[:,0:neffpoints])

   weighttot=np.sum(weightrep[:,0:neffpoints],axis=0)
for j in range(0,ndim):
   grad[0:neffpoints,j]=np.where(weighttot[0:neffpoints]>0,grad[0:neffpoints,j]/(weighttot[0:neffpoints]),0)

for i in range(0,neffpoints):
   with open("grad_combined.dat", 'a') as f:
       f.write("%s %s %s %s %s %s \n" % (i,colvarseff[i,0],colvarseff[i,1],grad[i,0],grad[i,1],weighttot[i]))

# BIN DATA (COULD BE NOT NECESSARY IF THE FREE ENERGY CAN BE RECONSTRUCTED USING A NON REGULAR GRID)
colvarsbineff=np.zeros((neffpoints,ndim))
nbins=1
diffc=np.zeros((neffpoints,ndim))
colvarsbineff[0,:]=colvarseff[0,:]
distance=np.zeros((neffpoints))
gradbin=np.zeros((neffpoints,ndim))
weightbin=np.zeros((neffpoints))
numbin=np.zeros((neffpoints),dtype=np.int32)
for i in range(0,neffpoints):
     diffc[i,:]=colvarseff[i,:]-colvarseff[0,:]
     diffc[i,:]=diffc[i,:]/box[0:ndim] 
     diffc[i,:]=diffc[i,:]-np.rint(diffc[i,:])*periodic[0:ndim]
     diffc[i,:]=diffc[i,:]*box[0:ndim]
     colvarbin=0.5*width+width*np.floor(diffc[i,:]/width)+colvarseff[0,:]
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
       nbins=nbins+1
       colvarsbineff[nbins-1,:]=colvarbin
     whichbin=np.argmin(distance[0:nbins])
     numbin[whichbin]=numbin[whichbin]+1
     gradbin[whichbin,:]=gradbin[whichbin,:]+grad[i,:]
     weightbin[whichbin]=weightbin[whichbin]+weighttot[i]

for j in range(0,ndim):
   gradbin[0:nbins,j]=np.where(numbin[0:nbins]>0,gradbin[0:nbins,j]/numbin[0:nbins],0)

for i in range(0,nbins):
   with open("grad_binned.dat", 'a') as f:
       f.write("%s %s %s %s %s \n" % (colvarsbineff[i,0],colvarsbineff[i,1],gradbin[i,0],gradbin[i,1],weightbin[i]))

              

     # now go on calculating distances 
   
# Function to calculate bias forces in a given point for a certain time

#def biasforce( colvarvector, time ):
#   "This prints a passed info into this function"
#   print "Name: ", name
#   print "Age ", age
#   return;

