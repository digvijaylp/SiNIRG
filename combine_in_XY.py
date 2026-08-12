import numpy as np
import sys
assert len(sys.argv)==3 # script-name, gro-file, n_reps

""""
Script for generating multiple replica file for
SARS-CoV spike-protein anchored to a XY-plane.

"""

del (sys.argv[0])  # delete script name
# single-replica grofile name
grofile=[x for x in sys.argv if x.endswith(".gro")][0]          
# number of replicas to add
nreps=int([x for x in sys.argv if not x.endswith(".gro")][0])

def parseGRO(infile):
    # load gro file data
    with open(infile) as fin:
        # loading number of atoms from line 2
        natoms=int([fin.readline() for x in range(2)][1])
        # loading grofile iines
        data=[fin.readline() for x in range(natoms)]
        # loading coordinates from lines
        XYZ=np.float64([[l[20:28],l[28:36],l[36:44]] for l in data])
        # removing coordinate and atom number data from lines
        data=[l[:15] for l in data]
    return data,XYZ
        
def align2Z(XYZ):
    # function to reorient the protein II to Z-axis
    # upper target CAs 293 from each chain of the trimer
    # above the membrane
    u_set=[293,709,1125]
    # lower target CAs 414 from each chain of the trimer
    # below the membrane
    l_set=[414,830,1246]

    # convert to XYZ indices (starting from 0)
    u_set=np.intp(u_set)-1
    l_set=np.intp(l_set)-1

    # centroids of the upper and lower triplets
    u_xyz=np.mean(XYZ[u_set],0)
    l_xyz=np.mean(XYZ[l_set],0)


    # vector along the two centroids
    spike_vec=u_xyz-l_xyz
    # unit vector along the two centroids
    u=spike_vec/(np.sum(spike_vec**2)**0.5)

    # cos and sin theta (theta with Z-axis)
    # u.[0,0,1]=u[2]
    cos_theta,sin_thtea=u[2],(1-u[2]**2)**0.5

    # normal to u and z-axis: rotation axis
    n=np.cross(u,[0,0,1])
    n=n/(np.sum(n**2)**0.5)
    # rotation matrix
    R=np.float64([[    0, -n[2],  n[1]],
                  [ n[2],     0, -n[0]],
                  [-n[1],  n[0],    0]])

    R=np.eye(3)+(sin_thtea*R)+((1-cos_theta)*np.dot(R,R))

    # new centroids of the upper and lower triplets
    reoriented_XYZ=np.dot(XYZ,R.T)
    u_xyz=np.mean(reoriented_XYZ[u_set],0)
    l_xyz=np.mean(reoriented_XYZ[l_set],0)

    # Z_ref 10.106 for the membrane
    # point for the 293 and 414 centroid 25.022,6.721 
    z_offset=np.float64([0,0,6.721])
    # re-center the trimer
    reoriented_XYZ=reoriented_XYZ-l_xyz+z_offset
    return reoriented_XYZ

def writeNRepGRO(outfile,lines,XYZ,N):
    # write N-rep gro file 
    # space b/w structures 
    str_gap= 20.0 # mm
    natoms=N*len(XYZ)
    grid_dim=int(np.ceil(N**0.5))
    # transform vector at the center of 2D cell # leaving 1 cell is buffer zone
    transvec=str_gap*np.float64([(x+2.5,y+2.5,0) for y in range(grid_dim) for x in range(grid_dim)])

    # write multiple replica file
    with open(outfile,"w+") as fout:
        # grofile first line
        fout.write("GRO with %d replicas.\n"%N)
        fout.write("%d\n"%natoms)
        atcount=0
        for n in range(N):
            # transforming coordinates for nth replica
            coords=XYZ+transvec[n]
            print ("Rep",n,transvec[n])
            for x in range(len(lines)):
                atcount+=1
                fout.write("%s%5d"%(lines[x],atcount%100000))
                fout.write("%8.3f%8.3f%8.3f\n"%tuple(coords[x]))
        # write GRO file Box line
        box_dim=(grid_dim+4)*str_gap
        # X and Y dimensions depend on number of replicas, Z dimension is fixed 
        fout.write("%8.3f%8.3f%8.3f\n"%(box_dim,box_dim,5*str_gap))
    return 1


# extracting data from gro file
data,XYZ=parseGRO(grofile)
# aligning transmembrane region to Z-axis
XYZ=align2Z(XYZ)
# write GRO file for N-replicas
outgro=grofile.replace(".gro",".%dnir.gro"%nreps)
writeNRepGRO(lines=data,XYZ=XYZ,N=nreps,outfile=outgro)