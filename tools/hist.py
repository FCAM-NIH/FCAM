import numpy as np
import argparse, os, sys
# now read file

def parse():
   parser=argparse.ArgumentParser()
   parser.add_argument("-tr", "--trajfile", \
                       help="input file containing the trajectory file containing: time, coordinate ", \
                       default="traj.in",type=str, required=True)
   parser.add_argument("-cvs","--cvlist", nargs="+",\
                       help="CVs to be analyzed", \
                       default=1,type=int,required=True)
   parser.add_argument("-grid","--grid", nargs="+",\
                       help="set grid size for each CV", \
                       default=100,type=int,required=True)
   parser.add_argument("-mincvs","--mincvs", nargs="+",\
                       help="set min val of each CV (othewise is found from the traj file)", \
                       type=float,required=False)
   parser.add_argument("-maxcvs","--maxcvs", nargs="+",\
                       help="set max val each CV (othewise is found from the traj file)", \
                       type=float,required=False)
   parser.add_argument("-out", "--outhistfile", \
                       help="output file containing the histogram: coordinate1, coordinate2, ..., counts  ", \
                       default="hist.out",type=str, required=False)
   parser.add_argument("-wcol", "--weightcolumn", \
                        help="specify weight column, if applicable. The weight is assumed to be in energy (or free energy, F) units and evaluated as a Boltzman factor (exp{-F/kbT}), unless -scalarweight is used", \
                        default=-1,type=int, required=False)
   parser.add_argument("-scalarweight","--scalarweight", \
                       help="by enabling this option the weight is considered as it is (scalar) and not a Boltzmann factor", \
                       default=False, dest='use_scalarw', action='store_true')
   parser.add_argument("-units", "--units", \
                       help="Choose free energy units specifying (case sensitive) either kj (kj/mol) or kcal (kcal/mol) (in alternative you can set the Boltzmann factor through the option -kb)", \
                       default="none",type=str, required=False)
   parser.add_argument("-kb", "--kb", help="Boltzmann factor for calculating the force constant (k) and defining free energy units.", \
                       default=-1,type=float, required=False)
   parser.add_argument("-temp", "--temp", help="Temperature (in Kelvin) for defining the free energy and integrating across CVs", \
                       default=-1,type=float, required=False)
   parser.add_argument("-fastbins","--fastbins", \
                       help=" use fast binning algorithm. Weights specification is not allowed in this modality and leads to an error", \
                       default=False, dest='use_fastb', action='store_true')


   args = parser.parse_args()
   return args

args=parse()
trajfile=args.trajfile
outhistfile=args.outhistfile
cvs=args.cvlist
grid=args.grid
ndim=len(cvs)
wcol=args.weightcolumn
temp=args.temp
units=args.units
kb=args.kb
use_fastb=args.use_fastb
use_scalarw=args.use_scalarw
if str(units)=="kj":
  kb=0.00831446261815324
elif str(units)=="kcal":
  kb=0.0019858775
elif kb<0:
    print ("ERROR: please specify either the units (-units) or the value of the Boltzmann factor (-kb option)")
    sys.exit()

if temp<=0:
    print ("ERROR: please specify a temperature (positive defined) to write the free energy in output and, if applicable, to calculate weights")
    sys.exit()


tmparray=np.loadtxt(trajfile)
numpoints=len(tmparray)
if wcol==-1:
  weights=np.ones((numpoints))
else:
  weights=tmparray[:,wcol]
  if use_scalarw==False:
    weights=np.exp(-weights/(kb*temp))
  else:
    negweights=np.where(weights<0)
    if len(negweights[0])>0:
      print ("ERROR: there are negative scalar weights. They should be all positive. If these are Boltzmannn factors please remove option -scalarweight")
      sys.exit()
      
  if use_fastb==True:
    print ("ERROR: weight specification is incompatible with fast binning algorithm")
    sys.exit()

assert(len(grid)==ndim)
if args.mincvs==None:
  mincvs=np.amin(tmparray[:,cvs],axis=0)
else:
  mincvs=args.mincvs
if args.maxcvs==None:
  maxcvs=np.amax(tmparray[:,cvs],axis=0)
else:
  maxcvs=args.maxcvs
assert(len(mincvs)==ndim)
assert(len(maxcvs)==ndim)

print("min values of CVs", mincvs)
print("max values of CVs", maxcvs)
print("grid values of CVs", grid)

binwidth=np.zeros((ndim))

for j in range(0,ndim):
   binwidth[j]=(maxcvs[j]-mincvs[j])/grid[j]

print("bin width of CVs", binwidth)

binarray=np.floor((tmparray[:,cvs]-mincvs)/binwidth)

if use_fastb==True:
  bingrid, counts=np.unique(binarray,axis=0,return_counts=True)
  nbins=len(counts)
  totcounts=np.sum(counts)
  maxweight=np.amax(counts)

  with open(outhistfile, 'w') as f:
      f.write("# colvar, probbability density \n")

  for i in range(0,nbins):
     with open(outhistfile, 'a') as f:
         for j in range(0,ndim):
            colvargrid=mincvs[j]+binwidth[j]*(0.5+bingrid[i,j]) 
            f.write("%s " % colvargrid) 
         f.write("%s " % (counts[i]/(totcounts*np.prod(binwidth))))
         f.write("%s \n " % (-kb*temp*np.log(counts[i]/(maxweight))))

# use simple loop
if use_fastb==False:
  bincounts=np.zeros((grid+np.ones(ndim,dtype=int)),dtype=np.float64)
  currentbin=np.zeros((ndim),dtype=np.int32)
  for i in range(0,numpoints):
     for j in range(0,ndim):
        currentbin[j]=int(binarray[i,j])
     bincounts[tuple(currentbin)]=bincounts[tuple(currentbin)]+weights[i]

  with open(outhistfile, 'w') as f:
      f.write("# colvar, probbability density \n")

  it = np.nditer(bincounts, flags=['multi_index'])
  totbincounts=np.sum(bincounts)
  maxweight=np.amax(bincounts)
  for x in it:
     with open(outhistfile, 'a') as f:
        if x>0:
          for j in range(0,ndim):
            colvargrid=mincvs[j]+binwidth[j]*(0.5+it.multi_index[j])
            f.write("%s " % colvargrid)
          f.write("%s " % (x/(totbincounts*np.prod(binwidth))))
          f.write("%s \n " % (-kb*temp*np.log(x/(maxweight))))
