import numpy as np
import sys
import glob
import math

def read_gro(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    title = lines[0].strip()
    natoms = int(lines[1].strip())
    atoms = lines[2:2+natoms]
    box = lines[2+natoms].strip()
    return title, natoms, atoms, box

def write_gro(filename, title, atoms, box="100.0  100.0  100.0"):
    with open(filename, 'w') as f:
        f.write(title + "\n")
        f.write(f"{len(atoms)}\n")
        for line in atoms:
            f.write(line)
        f.write(box + "\n")

def parse_atom_line(line):
    resid = line[0:5]
    resname = line[5:10]
    atomname = line[10:15]
    atomnum = line[15:20]
    x = float(line[20:28])
    y = float(line[28:36])
    z = float(line[36:44])
    rest = line[44:]
    return resid, resname, atomname.strip(), atomnum, x, y, z, rest

def format_atom_line(resid, resname, atomname, atomnum, x, y, z, rest):
    return f"{resid}{resname}{atomname:>5}{atomnum:5d}{x:8.3f}{y:8.3f}{z:8.3f}{rest}"

def generate_grid_positions(n, spacing=15.0):
    """
    Generate XY translation positions on a square grid.
    """
    ncols = math.ceil(math.sqrt(n))

    positions = []
    for i in range(n):
        row = i // ncols
        col = i % ncols
        dx = col * spacing
        dy = row * spacing
        positions.append((dx, dy))

    return positions

def recenter_atoms_xy(atoms, box_size=100.0):
    """Shift atoms so the system is centered in X and Y only."""
    
    xs, ys = [], []
    parsed = []

    for line in atoms:
        #print (line)
        resid, resname, atomname, atomnum, x, y, z, rest = parse_atom_line(line)
        xs.append(x)
        ys.append(y)
        parsed.append((resid, resname, atomname, atomnum, x, y, z, rest))

    # current center
    x_center = (min(xs) + max(xs)) / 2.0
    y_center = (min(ys) + max(ys)) / 2.0

    # box center
    target = box_size / 2.0

    dx = target - x_center
    dy = target - y_center

    recentered = []

    for resid, resname, atomname, atomnum, x, y, z, rest in parsed:
        recentered.append(
            format_atom_line(
                resid, resname, atomname, int(atomnum),
                x + dx, y + dy, z, rest   # Z unchanged
            )
        )

    return recentered

def combine_frames(N_frame=15,frame_files=[]):
    # list of input files
    initial_list=frame_files.copy()
    
    # concatinate the lists N_frame number of times
    while len(frame_files) < N_frame:
        frame_files += initial_list

    # get 2D grid positions
    positions = generate_grid_positions(len(frame_files), spacing=20.0)

    replica_atoms = []
    atom_counter = 1

    # looping over N_frame files to combine
    for i, frame_file in enumerate(frame_files):
        # load gro files
        title, natoms, atoms, _ = read_gro(frame_file)
        # get new x & y axis trasformation vec
        dx, dy = positions[i]
        # loop over atoms
        for line in atoms:
            # get line-wise paramters from grofile
            resid, resname, atomname, atomnum, x, y, z, rest = parse_atom_line(line)
            # transform co-ordinates
            x_new = x + dx
            y_new = y + dy
            # combine line-wise parameters and new coordinates as new grofile line
            atom_line = format_atom_line(
                resid, resname, atomname, atom_counter%100000,
                x_new, y_new, z, rest
            )
            # add lines to list 
            replica_atoms.append(atom_line)
            # new atom number counter
            atom_counter += 1

    # place replica centroid at grid cell center 
    replica_atoms = recenter_atoms_xy(replica_atoms, box_size=800.0)

    # write combined gro file
    write_gro(
        "opensmog.ghost%03dnir.gro"%N_frame,
        f"Combined {len(frame_files)} structures",
        replica_atoms,
        "800.0  800.0  800.0"
    )

if __name__ == "__main__":
    del sys.argv[0]     # 0th argument is the python script name
    # list of input files
    infiles=[x for x in sys.argv if x.endswith('.gro')]
    # load number of replicas from input argument
    ncopies=int([x for x in sys.argv if not x.endswith('.gro')][0])
    # combine frames
    combine_frames(N_frame=ncopies,frame_files=infiles)
