import numpy as np
from tqdm import tqdm,trange 
import argparse
from pathlib import Path
from joblib import Parallel, delayed    # for parallelizing PE calculations
from OpenSMOG import SBM                # for calculating single copy energy
from openmm import unit                 # for units 
import mdtraj as md                     # for loading and analysing xtc files 



"""
Code by Digvijay L. Prakash for decomposing energies from an 
OpenMM/OpenSMOG MD simulations with multiple non-interacting 
replicas. 

"""

# flags for input files
parser=argparse.ArgumentParser(description="Code for decomposing incidiviaul potential energies & Q from ghost runs. ")
parser.add_argument("-g","-s","-gro",type=str,help="Input single copy .gro file. Default=opensmog.gro",default="opensmog.gro")
parser.add_argument("-p","-top",type=str,help="Input single copy .gro file. Default=opensmog.top",default="opensmog.top")
parser.add_argument("-x","-xml",type=str,help="Input single copy .gro file. Default=opensmog.xml",default="opensmog.xml")
parser.add_argument("-c","-cmap",type=str,help="Input single copy cmap .CGcont file. Default=protCG.cont",default="protCG.cont")
parser.add_argument("-ng","-ngro",type=str,help="Input combined .gro file  (MD input). Default=None")
parser.add_argument("-f","-xtc",nargs='+',help="Input combined .xtc file  (MD output). Default=None",default=[])

# flags for analysis
parser.add_argument("-rg","-Rg",action="store_true",help="Save Radius of gyration values. Default: False ",default=False)
parser.add_argument("-natQ","-natq",type=int,help="Save native contact Q-values as (1) number or (2) fraction,Default: (0) No Q",default=0)
parser.add_argument("-edr","-PE",action="store_true",help="Save Potential Energy values. Default: False ",default=False)

# flags for Q calculations
parser.add_argument("-chi","-sop_chi",type=float,help="Calculate Q as structure-overlap-function using cutoff values in A. (Default: 0.0",default=0.0) 
parser.add_argument("-qc","-qcut",type=float,help="Q-scaling cutoff. Default=1.2",default=1.2)
parser.add_argument("-acmbins",type=int,help="Generate Contact map population files for N bins. Default=0",default=0)

# flags for PE calculations
parser.add_argument("-n","-nrep",type=int,help="Number of non-interacting replicas. Default 1",default=1)
parser.add_argument("-T","-temp",type=float,help="Temperature in reduced units. Default = 1 RU",default=1.00)
parser.add_argument("-dt","-step",type=float,help="Simulation step side reduced units or (ps). Default = 0.0005 RU",default=0.0005)
parser.add_argument("-rc","-rcut",type=float,help="r_cutoff. Default = 0.0005 nm",default=3.0)
parser.add_argument("-cpu",type=int,help="Use N CPU Default=4",default=4)

# other flags
parser.add_argument("-frames",type=int,help="Number of frames to be loaded at once. Default=200000",default=200000)

args=parser.parse_args()

# single copy files 
single_grofile=args.gro     # single-copy structure GRO file
single_topfile=args.top     # single-copy forcefield TOP file
single_xmlfile=args.xml     # single-copy forcefield XML file
single_cmapfile=args.cmap   # single-copy native contact cmap file

# single copy file checks

# grofile is required for loading trajectory.
assert Path(single_grofile).is_file(), 'Error. %s not found'%(single_grofile)
# get n_atoms in one structure from single-copy grofile
with open(single_grofile) as fin:
    n_atoms=int([fin.readline() for i in range(2)][-1])
if args.natQ>0:
    # contact Q calculations require single copy cmap file
    assert Path(single_cmapfile).is_file(), 'Error. %s not found'%(single_cmapfile)
    # load contact pairs and distances (convert distances to nm)
    cmap_pairs=-1+np.intp([l.split()[1:4:2] for l in open(single_cmapfile)])
    cmap_dists=0.1*np.float64([l.split()[5] for l in open(single_cmapfile)])
if args.edr:
    # energy calculations require single copy forcefield top and xml files
    assert Path(single_topfile).is_file(), 'Error. %s not found'%(single_topfile)
    assert Path(single_xmlfile).is_file(), 'Error. %s not found'%(single_xmlfile)
    # values for creating SBM object for energy re-calculations
    T=args.T
    dt=args.dt
    r_cutoff=args.rc
    q_cutoff=args.qc

# number of non-interacting replicas in the combined gro/xtc
n_reps=args.n                 
# loading combined gro file
if args.ngro: n_grofile=args.ngro
else: 
    # if combined gro file is not given, then n=1 and single copy gro file is used for loading xtc
    assert n_reps==1, "Error. n=1 is combined gro file not given as input"
    n_grofile=single_grofile
# combined.gro file necessary for loading trajectory.
assert Path(n_grofile).is_file(), 'Error. %s not found'%(n_grofile)

if args.chi>0:
    # chi cutoff for calculating Q as structure-overlap-function
    # natQ must be set to 1 or 2
    assert args.natQ>0


# initialize output file handlers for writing per replica values
if args.natQ:
    # per-replica Q values
    fQ={i:open("Q_rep%03d.dat"%i,"w+") for i in range(n_reps)}
if args.edr:
    # per-replica energy values
    fE={i:open("PE_rep%03d.dat"%i,"w+") for i in range(n_reps)}
    for i in range(n_reps): fE[i].close() #close immediately to avoid conflicts with parallel writing
if args.rg:
    # per-replica radius of gyration values
    fR={i:open("Rg_rep%03d.dat"%i,"w+") for i in range(n_reps)}

if args.acmbins:
    # create bins for average contact map population files
    # natQ must be set to 1 or 2 for calculating Q first
    assert args.natQ>0, "Error, Set -natQ 1 or 2 for calculating Q first"
    # number of bins
    n_bins=args.acmbins     
    # create bins for Q values
    Qbins=np.linspace(0,len(cmap_dists),args.acmbins)
    # map Q values from 0 to total contacts to corresponding bins
    Qbins=np.intp([np.where(x<Qbins)[0][0] for x in range(len(cmap_dists))])
    # create contact map histogram for each bin and each contact pair
    cmap_hist=np.zeros((args.acmbins,len(cmap_dists)),dtype=int)
    # create histogram for total frames in each bin
    total_frames=np.zeros(args.acmbins,dtype=int)

def Evaluate_PE(XYZ,index):
    # evaluate potential energy of each frame for a given replica
    # by loading single-copy forcefield in OpenMM/OpenSMOG 
    # and passing the coordinates of the replica

    # XYZ shape = [frames,n_atoms,3] 3:coordinates
    print (XYZ.shape)

    # create SBM object for single copy energy calculations
    sbm_1 = SBM(name="singleCopy", 
                time_step=dt, collision_rate=1.0, 
                r_cutoff=r_cutoff,temperature=T,
                pbc=False, cmm=True)

    # calculattion done CPU
    sbm_1.setup_openmm(platform="cpu")

    # load single copy structure and forcefield files 
    sbm_1.loadSystem(Grofile=single_grofile,
                     Topfile=single_topfile,
                     Xmlfile=single_xmlfile)
    # creating the simulation context for energy calculations
    sbm_1.createSimulation()

    # write potential energy values to file for the given replica
    with open("PE_rep%03d.dat"%index,"a") as fE:
        # loop over all frames for the given replica
        for j in trange(XYZ.shape[0],position=2,desc="Calculating PE of rep_%03d"%index):
            #passing co-ordinates to simulation context
            # XYZ[j] shape=[n_atoms,3] 3:coordinates
            sbm_1.simulation.context.setPositions(XYZ[j]*unit.nanometers)
            #calculating energy for the given coordinates
            state=sbm_1.simulation.context.getState(getEnergy=True)
            energy=state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            #writing PE-file
            fE.write("%.6f\n"%energy)
    return 0

# loop for all input xtc files
for xtcfile in args.xtc:
    assert xtcfile.endswith("xtc")

    # iteratively load trajectory in chunks of frames to avoid memory issues
    for traj in tqdm(md.iterload(xtcfile,top=n_grofile,chunk=args.frames),position=0,desc="Loading trajectory %s chunk "%xtcfile): 

        # load coordinates of all replicas in the chunk
        XYZ=traj.xyz  # shape=[frames,n_reps*n_atoms,3] 3:coordinates 
       
        # calculatee and write radius of gyration
        if args.rg:   
            # loop over all replicas
            for i in trange(n_reps,position=1,desc="Calculating replica Rad.of.Gyr.",leave=False):
                # load replica i atom indices
                rep_atoms=(n_atoms*i)+np.intp(list(range(0,n_atoms)))
                # caclulate replica i radius of gyration
                Rg=md.compute_rg(traj=traj.atom_slice(rep_atoms))
                # write replica i radius of gyration
                for j in range(len(Rg)):
                    fR[i].write("%.3f\n"%Rg[j])

        # calculate and write Q-values
        if args.natQ>0: 
            # loop over all replicas
            for i in trange(n_reps,position=1,desc="Calculating replica Q-values",leave=False):
                # load replica i native contact pairs atom indices
                pairs=(n_atoms*i)+cmap_pairs #increment indices by number of atoms

                # calculate distances for all replica i native contact pairs
                dists=md.compute_distances(traj=traj,periodic=True,atom_pairs=pairs)

                # create an number of frames x number of contacts array
                # of 1s and 0s for native contacts within cutoff distance
                # cutoff distance as q_cutoff * native contact distance
                Q=np.intp(dists<=q_cutoff*cmap_dists)
                if args.chi:
                    # cutoff distance as fixed value of chi in nm
                    Q=np.intp(0.1*args.chi>np.abs(dists-cmap_dists))
                # Q-values as number of native contacts within cutoff distance
                Q=np.sum(Q,1)

                # write contact map and population histograms
                if args.acmbins:
                    # increment total_frames array for each bin 
                    # corresponding to the Q-value of each frame
                    np.add.at(total_frames,(Qbins[Q]),1) 
                    # corresponding frame and contact pair indices
                    frame_idx,pair_idx=np.where(dists<=q_cutoff*cmap_dists)
                    # get binx for each contact pair in each frame
                    bin_idx=Qbins[Q[frame_idx]]
                    # increment the corresponding contact map histogram
                    np.add.at(cmap_hist,(bin_idx,pair_idx),1)

                if args.natQ==1:
                    #write Q value as number of native contacts
                    for j in range(len(Q)):
                        fQ[i].write("%d\n"%Q[j])
                elif args.natQ==2:
                    #write Q value as fraction of native contacts
                    Q = Q/len(cmap_dists)
                    for j in range(len(Q)):
                        fQ[i].write("%.6f\n"%Q[j])
        
        # calculate and write potential energy values
        if args.edr:  
            # split the coordinates of all replicas in the chunk 
            # into a list of arrays for parallel processing
            # XYZ shape=[frames,n_reps*n_atoms,3] 3:coordinates
            # split_xyz: list of n_rep arrays of shape=[frames,n_atoms,3]
            split_xyz=[]
            # looping over all replicas
            for i in range(n_reps): 
                # replica i atom indices 
                atoms=(n_atoms*i)+np.intp(range(n_atoms))
                # loading all frames of replica i
                sub_xyz=XYZ[:,atoms,:]
                # adding to list of arrays for parallel processing
                split_xyz.append(sub_xyz)
            # delete the large array to free memory
            del(XYZ) 
            # run Evaluate_PE function in parallel
            Parallel(n_jobs=args.cpu)(
                delayed(Evaluate_PE)(data,f_index) 
                for data,f_index in zip(split_xyz,range(n_reps))
                )

# write contact map and population files for each bin
if args.acmbins:
    with open('total_frames.mat',"w+") as fout:
        fout.write(str(total_frames))
    for j in range(1,args.acmbins):
        # ignore empty bins 
        if total_frames[j]==0: continue 
        # write fraction population of each contact pair (i,j and j,i) in the bin
        fN=open("Ncmap_%02d.dat"%j,"w+")
        for i in range(len(cmap_pairs)):
            fN.write("%d %d "%tuple(cmap_pairs[i]))
            fN.write("%e\n"%(cmap_hist[j][i]/total_frames[j]))
            fN.write("%d %d "%(cmap_pairs[i][1],cmap_pairs[i][0]))
            fN.write("%e\n"%(cmap_hist[j][i]/total_frames[j]))
        fN.close()

# close all output files
if args.natQ: 
    for i in range(n_reps): fQ[i].close()
if args.rg:
    for i in range(n_reps): fR[i].close() 