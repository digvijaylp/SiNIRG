from OpenSMOG import SBM
from openmm import CustomExternalForce
from openmm import unit
import argparse

# input argument flags 

# neccessary input arguments: -g, -p, -x, below
parser=argparse.ArgumentParser()
parser.add_argument("-g","-gro",type=str,help="Input structure GRO file.")
parser.add_argument("-p","-top",type=str,help="Input forcefield TOP file.")
parser.add_argument("-x","-xml",type=str,help="Input forcefield XML file.")

# optional input arguments below
parser.add_argument("-n","-nrep",type=int,help="Number of non-interacting replicas. Default 1",default=1)
parser.add_argument("-o","-out",type=str,help="Output folder name. Default=Out_01",default="Out_01")
parser.add_argument("-tempk",type=float,help="Simulation temperature in in K. Default=None")
parser.add_argument("-temp",type=float,help="Simulation temperature in reduced units. Default=1 RU",default=1.00)
parser.add_argument("-steps",type=int,help="Number of simulation steps. Default=100000000 steps",default=100000000)
parser.add_argument("-dt","-step",type=float,help="Simulation step size in reduced units or (ps). Default=0.0005 RU",default=0.0005)
parser.add_argument("-rt","-report",type=int,help="Reporting interval. Default=2000",default=2000)
parser.add_argument("-rc","-rcut",type=float,help="r_cutoff. Default=0.0005 nm",default=3.0)
parser.add_argument("-cr","-collisionrate",type=float,help="Collision rate. Default=1.0",default=1.0)
parser.add_argument("-op","-platform",type=str,help="OpenMM Simulation Platform",default="cuda")
args=parser.parse_args()

# OpenSMOG simulation parameters
simul_prefix="Output"   # Output file prefix
outputdir=args.o        # Output directory
dt=args.dt              # stepsize in redeuced time units (ps)
collision_rate=args.cr  # collision rate in inverse time units (ps-1)
r_cutoff=args.rc        # nonbond cutoff in nm
T=args.temp             # temperature reduced units
if args.tempk is not None:
    T_in_K=args.tempk           # temperature in Kelvin
    T=float(T_in_K)*0.008314    #reduced units RT

# creating SBM object with the above parameters
sbm_CA=SBM(name=simul_prefix, time_step=dt, collision_rate=collision_rate,
              r_cutoff=r_cutoff, temperature=T, pbc=True, cmm=False)
# setting up OpenMM simulation platform. 
# supported platform="cuda", "opencl", or "cpu" 
# cuda and opencl are GPU platforms, cpu is CPU platform.
platform=args.op
assert platform in ["cuda","opencl","cpu"],\
    "Please provide a valid OpenMM platform: cuda, opencl, or cpu."
sbm_CA.setup_openmm(platform=platform,GPUindex='default')

# loading the structure and forcefield files. 
assert args.g is not None, "Please provide a GRO file using -g option."
assert args.p is not None, "Please provide a TOP file using -p option."
assert args.x is not None, "Please provide a XML file using -x option."
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
reverse_flat_force.addPerParticleParameter("dummy")     # buffer paramter

# all paramter values for all replicas
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
report_interval=args.rt         # Reporting interval
sbm_CA.createReporters(
        trajectory=True,
        trajectoryFormat='xtc',
        energies=True,
        energy_components=True,
        interval=report_interval
)

# running the simulation
nsteps=int(args.steps)
sbm_CA.run(nsteps=nsteps, report=True, interval=report_interval)
# saving the final state and checkpoint files
sbm_CA.simulation.saveState("%s/endfile.state"%outputdir)
sbm_CA.simulation.saveCheckpoint("%s/endfile.chk"%outputdir)

