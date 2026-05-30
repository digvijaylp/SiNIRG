#from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
from OpenSMOG import SBM #for calculating single copy energy
from openmm import unit #for units 
import mdtraj as md #for loading and analysing xtc files 
import numpy as np #for numpy-ing 
import argparse #for input arguments 
from tqdm import tqdm,trange # because I like to look @ uber maps
from pathlib import Path

"""
Code by Digvijay L. Prakash for decomposing energies from an 
OpenMM/OpenSMOG MD simulations with multiple non-interacting 
replicates. 

"""

parser=argparse.ArgumentParser(description="Code for decomposing incidiviaul potential energies & Q from ghost runs. ")

parser.add_argument("-gro","-s",type=str,help="Input single copy .gro file. Default=opensmog.gro",default="opensmog.gro")
parser.add_argument("-top","-p",type=str,help="Input single copy .gro file. Default=opensmog.top",default="opensmog.top")
parser.add_argument("-xml","-x",type=str,help="Input single copy .gro file. Default=opensmog.xml",default="opensmog.xml")
parser.add_argument("-cmap","-c",type=str,help="Input single copy cmap .CGcont file. Default=protCG.cont",default="protCG.cont")
parser.add_argument("-dswap",action="store_true",help="Symmetrize contacts for domains-swapping simulations Default=False",default=False)
parser.add_argument("-ter",type=int,help="Last/Terminal atom of chain 1. Default=None",default=None)
parser.add_argument("-n","-nrep",type=int,help="Number of non-interacting replicates. Default 1",default=1)
parser.add_argument("-ngro","-ng","-g",type=str,help="Input combined .gro file  (MD input). Default=None")
parser.add_argument("-xtc","-f",nargs='+',help="Input combined .xtc file  (MD output). Default=None",default=[])
parser.add_argument("-T","-temp",type=float,help="Temperature in reduced units. Default = 1 RU",default=1.00)
parser.add_argument("-dt","-step",type=float,help="Simulation step side reduced units or (ps). Default = 0.0005 RU",default=0.0005)
parser.add_argument("-rc","-rcut",type=float,help="r_cutoff. Default = 0.0005 nm",default=3.0)
parser.add_argument("-edr","-PE",action="store_true",help="Save Potential Energy values. Default: False ",default=False)
parser.add_argument("-rg","-Rg",action="store_true",help="Save Radius of gyration values. Default: False ",default=False)
parser.add_argument("-natQ","-natq",type=int,help="Save frac.native contacts: Q-values as (1) number or (2) fraction,Default: No Q",default=0)
parser.add_argument("-chi","-sop_chi",type=float,help="Calculate Q as structure-overlap-function using cutoff values in A. (Default: 0.0",default=0.0) 
parser.add_argument("-qc","-qcut",type=float,help="Q-scaling cutoff. Default=1.2",default=1.2)
parser.add_argument("-acmbins",type=int,help="Generate Contact map population files for N bins. Default=0",default=0)
parser.add_argument("-frames",type=int,help="Number of frames to be loaded at once. Default=200000",default=200000)
parser.add_argument("-time",action="store_true",help="Write time in the output file. Default=False",default=False)
parser.add_argument("-cpu",type=int,help="Use N CPU Default=4",default=4)
args=parser.parse_args()

#single copy files 
single_grofile=args.gro
single_topfile=args.top
single_xmlfile=args.xml
single_cmapfile=args.cmap

assert Path(single_grofile).is_file(), 'Error. %s not found'%(single_grofile)
if args.edr:
    assert Path(single_topfile).is_file(), 'Error. %s not found'%(single_topfile)
    assert Path(single_xmlfile).is_file(), 'Error. %s not found'%(single_xmlfile)
if args.natQ>0:
    assert Path(single_cmapfile).is_file(), 'Error. %s not found'%(single_cmapfile)

#combined files 
n_replicates=args.n
if args.ngro: n_grofile=args.ngro
else: 
    n_grofile=single_grofile
    assert n_replicates==1, "Error. n=1 is combined gro file not given as input"
assert Path(n_grofile).is_file(), 'Error. %s not found'%(n_grofile)

#dymmy values for creating SBM object 
T=args.T
dt=args.dt
r_cutoff=args.rc
q_cutoff=args.qc

#get n_atoms in one structure from grofile
with open(single_grofile) as fin:
    n_atoms=int([fin.readline() for i in range(2)][-1])

if args.chi>0: assert args.natQ>0
if args.natQ>0:
    #load contact pairs and distances
    cmap_pairs=-1+np.intp([l.split()[1:4:2] for l in open(single_cmapfile)])
    cmap_dists=0.1*np.float64([l.split()[5] for l in open(single_cmapfile)])

if args.ter or args.dswap:
    if args.dswap:
        if not args.ter: args.ter=n_atoms//2
        ter=args.ter
        p_c1c1=cmap_pairs.copy()
        p_c1c2=p_c1c1+np.intp([0,ter])
        p_c2c1=p_c1c1+np.intp([ter,0])
        p_c2c2=p_c1c1+ter
        d_c1c1,d_c2c2,d_c1c2,d_c2c1=cmap_dists.copy(),cmap_dists.copy(),cmap_dists.copy(),cmap_dists.copy()
    else:
        assert args.ter
        ter=args.ter
        p_c1c1,p_c2c2,p_c1c2,p_c2c1=[],[],[],[]
        d_c1c1,d_c2c2,d_c1c2,d_c2c1=[],[],[],[]
        for i in range(len(cmap_dists)):
            x,y=cmap_pairs[i]
            if x<=ter and y<=ter:
                p_c1c1.append((x,y))
                d_c1c1.append(cmap_dists[i])
            elif x>ter and y>ter: 
                p_c2c2.append((x,y))
                d_c2c2.append(cmap_dists[i])
            else: 
                if x<=ter: 
                    assert y>ter
                    p_c1c2.append((x,y))
                    d_c1c2.append(cmap_dists[i])
                elif y<=ter:
                    assert x>ter
                    p_c2c1.append((x,y))
                    d_c2c1.append(cmap_dists[i])

    cmap_pairs=np.concat([l for l in (p_c1c1,p_c2c2,p_c1c2,p_c2c1) if len(l)!=0])
    cmap_dists=np.concat([l for l in (d_c1c1,d_c2c2,d_c1c2,d_c2c1) if len(l)!=0])

    is_intra1=np.intp([x<len(p_c1c1) for x in range(len(cmap_pairs))])
    is_intra2=np.intp([(x>=len(p_c1c1) and x<len(p_c1c1)+len(p_c2c2))\
                                 for x in range(len(cmap_pairs))])
    is_inter=np.intp([x>=len(p_c1c1)+len(p_c2c2) for x in range(len(cmap_pairs))])
    assert np.sum(is_intra1)==len(p_c1c1)
    assert np.sum(is_intra2)==len(p_c2c2)
    assert np.sum(is_inter)==len(p_c1c2)+len(p_c2c1)

if args.natQ:
    fQ={i:open("Q_rep%03d.dat"%i,"w+") for i in range(n_replicates)}
    if args.ter:
        fI={i:open("2dQ_rep%03d.dat"%i,"w+") for i in range(n_replicates)}
if args.edr:
    fE={i:open("PE_rep%03d.dat"%i,"w+") for i in range(n_replicates)}
    for i in range(n_replicates): fE[i].close() #close immediately to avoid conflicts with parallel writing
if args.rg:
    fR={i:open("Rg_rep%03d.dat"%i,"w+") for i in range(n_replicates)}


if args.acmbins:
    n_bins=args.acmbins
    assert args.natQ>0, "Error, Set -natQ 1 or 2 for calculating Q first"
    if not args.ter:
        Qbins=np.linspace(0,len(cmap_dists),args.acmbins)
        Qbins=np.intp([np.where(x<Qbins)[0][0] for x in range(len(cmap_dists))])
        cmap_hist=np.zeros((args.acmbins,len(cmap_dists)),dtype=int)
        total_frames=np.zeros(args.acmbins,dtype=int)
        #print (cmap_hist.shape,cmap_hist[0].shape,total_frames.shape)
        #cmap_hist={(i,j):np.zeros(len(cmap_dists)) for j in range(1,args.acmbins) for i in range(n_replicates)}
    elif args.ter:
        n_intra=len(d_c1c1)+len(d_c2c2)
        n_inter=len(d_c1c2)+len(d_c2c1)
        intra_bins=np.linspace(0,n_intra,n_bins)
        intra_bins=np.intp([np.where(x<intra_bins)[0][0] for x in range(n_intra)])
        inter_bins=np.linspace(0,n_inter,n_bins)
        inter_bins=np.intp([np.where(x<inter_bins)[0][0] for x in range(len(d_c1c2)+len(d_c2c1))])
        cmap_hist=np.zeros((n_bins,n_bins,len(cmap_dists)),dtype=int)
        total_frames=np.zeros((n_bins,n_bins),dtype=int)

def Evaluate(XYZ,index):
    print (XYZ.shape)
    sbm_1 = SBM(name="singleCopy", time_step=dt, collision_rate=1.0, r_cutoff=r_cutoff, temperature=T, pbc=True, cmm=True)
    sbm_1.setup_openmm(platform="cpu")
    sbm_1.loadSystem(Grofile=single_grofile, Topfile=single_topfile, Xmlfile=single_xmlfile)
    sbm_1.createSimulation()
    with open("PE_rep%03d.dat"%index,"a") as fE:
        #for j in trange(XYZ.shape[0],position=2,desc="Calculating PE",leave=False):
        for j in trange(XYZ.shape[0],position=2,desc="Calculating PE of rep_%03d"%index):
            #giving co-ordinates
            sbm_1.simulation.context.setPositions(XYZ[j]*unit.nanometers)
            #calculating energy
            state=sbm_1.simulation.context.getState(getEnergy=True)
            energy=state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            #writing PE-file
            #if args.time: fE[i].write("%e "%TIME[j])
            fE.write("%.6f\n"%energy)

for xtcfile in args.xtc:
    assert xtcfile.endswith("xtc")
    #traj=md.load(xtcfile,top=n_grofile) #,atom_indices=atoms)
    for traj in tqdm(md.iterload(xtcfile,top=n_grofile,chunk=args.frames),position=0,desc="Loading trajectory %s chunk "%xtcfile): #,atom_indices=atoms)
        XYZ=traj.xyz #shape=[frames,n_atoms,3] 3:coordinates 
        TIME=traj.time #shape=[frames,1] 1:time step
        if args.rg: #write Rg
            for i in trange(n_replicates,position=1,desc="Calculating Replicate Rad.of.Gyr.",leave=False):
                rep_atoms=(n_atoms*i)+np.intp(list(range(0,n_atoms)))
                Rg=md.compute_rg(traj=traj.atom_slice(rep_atoms))
                for j in range(len(Rg)):
                    if args.time:
                        fR[i].write("%e "%TIME[j])
                    fR[i].write("%.3f\n"%Rg[j])
        if args.natQ>0: #write Q
            for i in trange(n_replicates,position=1,desc="Calculating Replicate Q-values",leave=False):
                #print (">>> Calculating rep%03d Q-values."%i)
                pairs=(n_atoms*i)+cmap_pairs #increment indices by number of atoms
                dists=md.compute_distances(traj=traj,periodic=True,atom_pairs=pairs)
                Q=np.intp(dists<=q_cutoff*cmap_dists)
                if args.chi:
                    Q=np.intp(0.1*args.chi>np.abs(dists-cmap_dists))
                if args.ter:
                    Q1=np.sum(Q*is_intra1,1)
                    Q2=np.sum(Q*is_intra2,1)
                    QM=Q1+Q2 #Q-monomeric
                    QD=np.sum(Q*is_inter,1) #Q-dimeric
                Q=np.sum(Q,1)
                #if args.ter:
                #   for i in range(len(Q)): assert (Q[i]==Q1[i]+Q2[i]+QI[i])
                if args.acmbins:
                    if not args.ter:
                        #for x in Qbins[Q]: total_frames[x]+=1
                        np.add.at(total_frames,(Qbins[Q]),1) #same as above
                        frame_idx,pair_idx=np.where(dists<=q_cutoff*cmap_dists)
                        bin_idx=Qbins[Q[frame_idx]]
                        #cmap_hist[bin_idx,pair_idx] # increments mult. entr. once
                        np.add.at(cmap_hist,(bin_idx,pair_idx),1)
                    else:
                        np.add.at(total_frames,(intra_bins[QM],inter_bins[QD]),1)  #increment
                        frame_idx,pair_idx=np.where(dists<=1.2*cmap_dists)
                        bin_idx_m=intra_bins[QM][frame_idx]
                        bin_idx_d=inter_bins[QD][frame_idx]
                        np.add.at(cmap_hist,(bin_idx_m,bin_idx_d,pair_idx),1)
                #writing Q-file
                if args.natQ==1:
                    for j in range(len(Q)):
                        if args.time: 
                            fQ[i].write("%e "%TIME[j])
                            if args.ter: fI[i].write("%e "%TIME[j])
                        fQ[i].write("%d\n"%Q[j])
                        if args.ter: fI[i].write("%d %d %d\n"%(Q1[j],Q2[j],QD[j]))
                elif args.natQ==2: #write fraction Q
                    Q /= len(cmap_dists)
                    if args.ter:
                        Q1 /= np.sum(is_intra1)
                        Q2 /= np.sum(is_inter)
                    for j in range(len(Q)):
                        if args.time: 
                            fQ[i].write("%e "%TIME[j])
                            if args.ter: fI[i].write("%e "%TIME[j])
                        fQ[i].write("%.6f\n"%Q[j])
                        if args.ter: fI[i].write("%d %d %d\n"%(Q1[j],Q2[j],QD[j]))
        if args.edr:
            split_xyz=[]
            for i in range(n_replicates):
                atoms=(n_atoms*i)+np.intp(range(n_atoms))
                sub_xyz=XYZ[:,atoms,:]
                split_xyz.append(sub_xyz)
                #brdcst_xyz = np.broadcast_to(sub_xyz[np.newaxis, :], (25, 20000, 258, 3))
            del(XYZ)
            Parallel(n_jobs=args.cpu)(
                            delayed(Evaluate)(data,f_index) 
                            for data,f_index in zip(split_xyz,range(n_replicates))
                            )
            #with ProcessPoolExecutor(max_workers=25) as executor:
            #    executor.map(Evaluate,brdcst_xyz,fE)

if args.acmbins:
    #fN={(i,j):open("Ncmap_%d_rep%03d.xvg"%(j,i),"w+") for j in range(args.acmbins) for i in range(n_replicates)}    
    with open('total_frames.mat',"w+") as fout:
        fout.write(str(total_frames))
    print (total_frames)
    if not args.ter:
        for j in range(1,args.acmbins):
            if total_frames[j]==0: continue
            fN=open("Ncmap_%02d.dat"%j,"w+")
            for i in range(len(cmap_pairs)):
                fN.write("%d %d "%tuple(cmap_pairs[i]))
                fN.write("%e\n"%(cmap_hist[j][i]/total_frames[j]))
                fN.write("%d %d "%(cmap_pairs[i][1],cmap_pairs[i][0]))
                fN.write("%e\n"%(cmap_hist[j][i]/total_frames[j]))
            fN.close()
    else:
        for j in range(1,args.acmbins):
            for k in range(1,args.acmbins):
                if total_frames[j,k]==0: continue
                fN=open("Ncmap_%02d_%02d_%d.dat"%(j,k,total_frames[j,k]),"w+")
                for i in range(len(cmap_pairs)):
                    fN.write("%d %d "%tuple(cmap_pairs[i]))
                    fN.write("%e\n"%(cmap_hist[j,k][i]/total_frames[j,k]))
                    fN.write("%d %d "%(cmap_pairs[i][1],cmap_pairs[i][0]))
                    fN.write("%e\n"%(cmap_hist[j,k][i]/total_frames[j,k]))
                fN.close()            
if args.natQ:
    for i in range(n_replicates): fQ[i].close()
    if args.ter:
        for i in range(n_replicates): fI[i].close()
if args.rg:
    for i in range(n_replicates): fR[i].close() #close immediately to avoid conflicts with parallel writing

exit()
