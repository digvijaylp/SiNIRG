
from OpenSMOG import SBM
from openmm import CustomExternalForce
from openmm import unit
import argparse

"""
Script for performing Spike protein CA-SBM simulations on OpenSMOG 
with inverse-flat-bottom restrain

"""

# input argument flags 
parser=argparse.ArgumentParser()

# neccessary input arguments: -g, -p, -x, below
parser.add_argument("-g","-gro",type=str,help="Input structure GRO file.")
parser.add_argument("-p","-top",type=str,help="Input forcefield TOP file.")
parser.add_argument("-x","-xml",type=str,help="Input forcefield XML file.")

# optional input arguments below
parser.add_argument("-n","-nrep",type=int,
help="Number of non-interacting replicas. Default 1",default=1)
parser.add_argument("-o","-out",type=str,
help="Output folder name. Default=Output_01",default="Output_01")
parser.add_argument("-steps",type=int,
help="Number of simulation steps. Default=100000000 steps",
default=100000000)
parser.add_argument("-op","-platform",type=str,
      help="OpenMM Simulation Platform",default="cuda")
args=parser.parse_args()

# OpenSMOG simulation parameters
prefix="Output"		# Output file prefix
outputdir=args.o        # Output directory
dt=0.0005			# step size in reduced time units (ps)
collision_rate=1.0	# collision rate in inverse time units (ps-1)
r_cutoff=3.0		# nonbond cutoff in nm
T=0.75			# temperature reduced units

# creating SBM object with the above parameters
sbm_CA=SBM(name=prefix, time_step=dt, collision_rate=collision_rate,
              r_cutoff=r_cutoff, temperature=T, pbc=True, cmm=False)

# setting up OpenMM simulation platform. 
# supported platform="cuda", "opencl", or "cpu" 
# cuda and opencl are GPU platforms, cpu is CPU platform.
platform= args.op
sbm_CA.setup_openmm(platform=platform,GPUindex='default')

# loading the structure and forcefield files. 
assert args.g is not None, "Please provide GRO file using -g option."
assert args.p is not None, "Please provide TOP file using -p option."
assert args.x is not None, "Please provide XML file using -x option."
sbm_CA_grofile=args.g   # GRO file
sbm_CA_topfile=args.p   # TOP file
sbm_CA_xmlfile=args.x   # XML file
sbm_CA.loadSystem(Grofile=sbm_CA_grofile,
                  Topfile=sbm_CA_topfile,
                  Xmlfile=sbm_CA_xmlfile)

# adding restraints along z-axis. restraining in XY-plane
n_replicas=args.n       # number of reps
atoms_per_replica=1248  # atoms in one replica

# original restraint ranges for one replica (0-based indexing)
restraint_ranges_3=[(235, 293), (651, 709), (1067, 1125),
                    (329, 374), (745, 790), (1161, 1206),
                    (414,415), (830,831), (1246,1247)]

# defining reverse flat-bottom potential (keep away from membrane zone)
# using CustomExternalForce() method imported from OpenMM
reverse_flat_force=CustomExternalForce("""
step(z_tol2 -abs(z - z_ref3) ) * 0.5 * k * (z_tol2 - abs(z - z_ref3))^2
""")
reverse_flat_force.addGlobalParameter("z_ref3", 10.106) # lower bound in nm
reverse_flat_force.addGlobalParameter("z_tol2", 2.5)    # tolerance
reverse_flat_force.addGlobalParameter("k", 10)          # force constant
reverse_flat_force.addPerParticleParameter("dummy")     # buffer parameter

# all parameter values for all replicas
for replica_index in range(n_replicas):
    offset=replica_index*atoms_per_replica
    for start, end in restraint_ranges_3:
        for atom_index in range(start, end + 1):
            reverse_flat_force.addParticle(atom_index + offset, [0.0])
# add force to the SBM object
sbm_CA.system.addForce(reverse_flat_force)

# initialize simulation context
sbm_CA.createSimulation()

# preparing the output directory and reporters
sbm_CA.saveFolder(outputdir)    # Creating the output folder
report_interval=2000            # Reporting interval
sbm_CA.createReporters(
        trajectory=True,
        trajectoryFormat='xtc',
        energies=True,
        energy_components=True,
        interval=report_interval)

# running the simulation
nsteps=int(args.steps)
sbm_CA.run(nsteps=nsteps, report=True, interval=report_interval)

# saving the final state and checkpoint files
sbm_CA.simulation.saveState("%s/endfile.state"%outputdir)
sbm_CA.simulation.saveCheckpoint("%s/endfile.chk"%outputdir)


