import mdtraj as md
from openmm import unit
from OpenSMOG import SBM
import os
import numpy as np
import argparse
import random as rnd

# input argument flags 
parser=argparse.ArgumentParser()
parser.add_argument("-g","-gro",type=str,help="Input structure GRO file.")
parser.add_argument("-p","-top",type=str,help="Input forcefield TOP file.")
parser.add_argument("-x","-xml",type=str,help="Input forcefield XML file.")
parser.add_argument("-c","-cmap",type=str,help="Input SuBMIT format contact map file.")
parser.add_argument("-n","-nrep",type=int,help="Number of non-interacting replicas. Default 1",default=1)
parser.add_argument("-steps",type=int,help="Number of simulations steps. Default=10000000",default=10000000)
args=parser.parse_args()

def simAnnealing(n_rep):
    # run simulated annealing to generate non-interacting replicas with different Q values.

    # simulation paramters
    outinterval=2000    #xtc and energy output interval 
    T_in_K=300          # initial temperature K
    R=0.008314          # KJ/mol
    T=float(T_in_K)*R   # reduced units RT  
    dt=0.0005           # stepsize in redeuced time units (ps)
    colrate=1.0        # collision rate in inverse time units (ps-1)
    r_cut=3.0           # nonbond cutoff in nm

    # create SBM object 
    sbm_1=SBM(name="Output", time_step=dt, collision_rate=colrate,
              r_cutoff=r_cut, temperature=T, pbc=True, cmm=True)

    # loading the structure and forcefield files.
    assert args.g is not None, "Please provide a GRO file using -g option."
    assert args.p is not None, "Please provide a TOP file using -p option."
    assert args.x is not None, "Please provide a XML file using -x option."
    assert args.c is not None, "Please provide a SuBMIT format contact map file using -c option."
    sbm_1_grofile=args.gro      # GRO file
    sbm_1_topfile=args.top      # TOP file
    sbm_1_xmlfile=args.xml      # XML file
    sbm_1_cmapfile=args.cmap    # cont file
    
    # setting up OpenMM simulation platform.
    # supported platform="cuda", "opencl", or "cpu"
    # cuda and opencl are GPU platforms, cpu is CPU platform.
    sbm_1.setup_openmm(platform="opencl", GPUindex="default")
    sbm_1.loadSystem(Grofile=sbm_1_grofile,
                     Topfile=sbm_1_topfile,
                     Xmlfile=sbm_1_xmlfile)
    # initialize the simulation context
    sbm_1.createSimulation()
    # preparing the output directory and reporters
    sbm_1.saveFolder("SimAn_00")
    sbm_1.createReporters(trajectory=True, trajectoryFormat="xtc",
                    energies=True, energy_components=True,
                    interval=outinterval)

    # simulated annealing parameters
    # total annealing cycles
    n_cycles=5  
    # 1 cycle: temperature schedule and corresponding step at each temperature
    temperatures=[300,200,125,110,100,95,90,85]
    step_list=[12500,12500,25000,25000,50000,50000,100000,100000]
    # the above list is for 1 cycle, and repeated n_cycles times.

    # running simulated annealing cycles
    # looping over cycles
    for i in range(n_cycles):
        # looping over temperatures in each cycle
        # controlled cooling of the system from high to low temperature
        for j in range(len(temperatures)):
            print ("Cycle %d temperature %.2f"%(i,temperatures[j]))
            T_in_K,n_steps=temperatures[j],step_list[j]
            # setting new simulation temperature and velocities
            sbm_1.simulation.integrator.setTemperature(T_in_K*unit.kelvin)
            sbm_1.simulation.context.setVelocitiesToTemperature(T_in_K*unit.kelvin)
            # running steps at the current temperature
            sbm_1.run(nsteps=n_steps, report=True, interval=2000)

    # get n_rep differnet Q values for each of the n_rep different replicas
    Q0=np.linspace(0.2,0.8,n_rep)

    # loading contact pairs and distances (and converting to nm)
    cmap_pairs=-1+np.intp([l.split()[1:4:2] for l in open(sbm_1_cmapfile)])
    cmap_dists=0.1*np.float64([l.split()[5] for l in open(sbm_1_cmapfile)])

    # loading the simulated annealing trajectory
    traj=md.load("SimAn_00/Output_trajectory.xtc",top=sbm_1_grofile)
    tstep=traj.time     # time array
    XYZ=traj.xyz        # coordinates array
    topol=traj.topology # topology object

    # compute distances for native contact pairs
    dist=md.compute_distances(traj=traj,periodic=True,atom_pairs=cmap_pairs)
    # finding total contacts with distance within the cutoff distance (1.2*cmap_dists) 
    Q=np.sum(np.intp(dist<=1.2*cmap_dists),1)/len(cmap_pairs)
    # saving the Q values to a file
    with open("SimAn.Q","w+") as fout:
        for i in Q: fout.write("%e\n"%i)
    # classifying structures into Q0 bins
    # bin margin for each Q0 value
    margin=0.5*(Q0[1]-Q0[0]) 
    # Use Q values to add indicies of structures to their corresponding Q0 bin.
    str_indices=[]
    for q in Q0:
        str_indices.append(np.where(np.abs(Q-q)<=margin)[0])

    # select 3 random structures form each bin
    random_str=[]   # list of randome structure for each Q0 bin
    n_rnd_str=3     # number of random structures per Q0 bin

    # writing population of structure in each Q0 bin
    with open("SimAn_Q0_hist.xvg","w+") as fout:
        population=0
        for i in range(n_rep):
            fout.write("%f %d\n"%(Q0[i],len(str_indices[i])))
            population+=len(str_indices[i])
            # load 3 random structures from each Q0 bin
            random_str.append(rnd.choices(str_indices[i],k=n_rnd_str))

    # setting up the non-interacting replicas in a grid 

    # number of grid cells per dimension (for 3D cubinc setup)
    # minimum cells required = number of replicas
    # actual number of cells is lowest cube number required
    grid_dim=int(np.ceil(n_rep**(1/3)))

    # width of each grid cell
    grid_width=50.0

    # transformation vector to put each replica in its grid cell
    trans_vec=grid_width*np.intp([(x,y,z)\
                        for z in range(grid_dim)\
                        for y in range(grid_dim)\
                        for x in range(grid_dim)])
    trans_vec += grid_width 
    
    # buffer zone in case of pbc
    pbc_box_dim=np.max(trans_vec)+grid_width

    # combinding sinle-copy mdtraj topology object to n_rep object
    combined_topol=topol
    for x in range(n_rep-1):
        combined_topol=combined_topol.join(topol)

    # creating 3 empty list for holding n_rep replica coordinates per list
    combined_xyz=[list() for i in range(n_rnd_str)]

    # looping over replica indices
    for i in range(n_rep):
        # looping over number of sets of random structures (= 3) 
        # generating 3 non-interacting replica strcuture files 
        for j in range(n_rnd_str):
            index=random_str[i][j]  # structure index 
            rep_xyz=XYZ[index]      # structure coordinate
            # move replic coordinates centroid to origin [0,0,0]
            rep_xyz-=np.mean(rep_xyz,0) 
            # create replica mdtraj trajectory object
            rep_trj=md.Trajectory(xyz=rep_xyz+grid_width/2,topology=topol)
            # define replica box unit cell dimensions
            rep_trj.unitcell_lengths=np.array([grid_width]*3)
            rep_trj.unitcell_angles=np.array([90.0]*3)        
            # save per replica structure
            rep_trj.save_gro("SimAn_00/rep%03d_%d.gro"%(i,j))
            print ("Generate SimAn_00/rep%03d_%d.gro"%(i,j))
            # move replica coordinate to its grid cell centre
            # use transormation vector
            # this will be later combined as single multiple replica file 
            rep_xyz+=trans_vec[i]
            combined_xyz[j].append(rep_xyz)
            # clear memory 
            del (rep_xyz,rep_trj)

    # looping over number of structure files to be generated (3)
    for j in range(n_rnd_str):
        # concatinating transformed coordinates of each replica
        combined_xyz[j]=np.concatenate([combined_xyz[j][i] for i in range(n_rep)])
        # creating n_rep mdtraj trajectory object using combined topology
        combined_trj=md.Trajectory(xyz=combined_xyz[j],topology=combined_topol)
        # defining box dimensions 
        combined_trj.unitcell_lengths=np.array([pbc_box_dim]*3)
        combined_trj.unitcell_angles=np.array([90.0]*3)        
        # output file name with file index (1/3)
        combined_grofile=".".join(sbm_1_grofile.split(".")[:-1]+["ghost%03dnir_%d"%(n_rep,j)]+["gro"])
        print (combined_grofile,combined_xyz[j].shape)
        # saving n_rep replicas combined structure file
        combined_trj.save_gro(combined_grofile)
    return 0

print ("> Generating multi-non-interacting-replica-gro file with different Q values.")
n_grofile=simAnnealing(n_rep=args.n_rep)