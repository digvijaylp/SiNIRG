
from OpenSMOG import SBM
import argparse

"""
Script for performing CA-SBM simulations on OpenSMOG 

"""
# input argument flags 
# neccessary input arguments: -g, -p, -x, below
parser=argparse.ArgumentParser()
parser.add_argument("-g","-gro",type=str,help="Input structure GRO file.")
parser.add_argument("-p","-top",type=str,help="Input forcefield TOP file.")
parser.add_argument("-x","-xml",type=str,help="Input forcefield XML file.")

# optional input arguments below
parser.add_argument("-o","-out",type=str,
            help="Output folder name. Default = Out_01",
            default="Output_01")
parser.add_argument("-temp",type=float,
            help="Simulation temperature in reduced units. Default = 1 RU",
            default=1.00)
parser.add_argument("-steps",type=int,
            help="Number of simulation steps. Default=100000000 steps", 
            default=100000000)
parser.add_argument("-dt","-step",type=float,
          help="Simulation step size in reduced units. Default = 0.0005 RU",
          default=0.0005)
parser.add_argument("-rt","-report",type=int,
            help="Reporting interval. Default = 2000",default=2000)
parser.add_argument("-rc","-rcut",type=float,
            help="r_cutoff. Default = 0.0005 nm",default=3.0)
parser.add_argument("-cr","-collisionrate",type=float,
            help="Collision rate. Default = 1.0",default=1.0)
parser.add_argument("-op","-platform",type=str,
            help="OpenMM Simulation Platform",default="cuda")
args=parser.parse_args()

# OpenSMOG simulation parameters
simul_prefix="Output"   # Output file prefix
outputdir=args.o        # Output directory
dt=args.dt              # stepsize in redeuced time units (ps)
collision_rate=args.cr  # collision rate in inverse time units (ps-1)
r_cutoff=args.rc        # nonbond cutoff in nm
T=args.temp             # temperature reduced units

# creating SBM object with the above parameters
sbm_CA = SBM(name=simul_prefix, time_step=dt, collision_rate=collision_rate,
              r_cutoff=r_cutoff, temperature=T,pbc=True)

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

# initialize simulation context
sbm_CA.createSimulation()
# preparing the output directory and reporters
sbm_CA.saveFolder(outputdir)    # Creating the output folder
trjformat="xtc"                 # Output trajectory format.
report_interval=args.rt         # Reporting interval
sbm_CA.createReporters(trajectory=True,trajectoryFormat=trjformat,
                       energies=True,energy_components=True,
                       interval=report_interval)    # Reporters 

# running the simulation
nsteps=int(args.steps)
sbm_CA.run(nsteps=nsteps, report=True, interval=report_interval)
# saving the final state and checkpoint files
sbm_CA.simulation.saveState("%s/endfile.state"%outputdir)
sbm_CA.simulation.saveCheckpoint("%s/endfile.chk"%outputdir)


