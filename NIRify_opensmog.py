import numpy as np
import sys
import argparse

"""
Script for converting N replica SBM forcefield file to NIR file 

"""

# flags for input files
parser=argparse.ArgumentParser(description="Code for converting SuBMIT generated N-replica forcefield files to N-non-interacting-replicas forcefeild files")

parser.add_argument("-ff",nargs='+',type=str,help=\
"Input N-replica forcefield .top and/or .xml file (SuBMIT generated).")
parser.add_argument("-gro",type=str,help="Input N-replica .gro file.")
parser.add_argument("-n","-nrep",type=int,\
help="Number of replicas. Default 1",default=1)
args=parser.parse_args()

# checking input arguments 
assert args.gro, "Please provide .gro file using -gro flag"
assert args.ff,  "Please provide .top and/or .xml files using -ff flag"
assert args.n,  "Please provide number of replicas using -n flag"
grofile=args.gro            # structure .gro file
infiles=args.ff             # forcefield .top and/or .xml file
nrep=args.n                 # number of replicas

# load structure file to get total and per-replica number of atoms
with open(grofile) as fin:
    # total atoms in nrep replica
    natoms=int([fin.readline() for i in range(2)][1])
    # atoms in 1 replica
    natoms_per_rep=int(natoms/nrep)

rep_index=0           # replica counter
atomtypes=[]          # list of new replica-indexed atomtypes 
CX_C12=[]             # attractive (CX) and repulsive (C12) nonbond params

# forcefield file extension list
filetypes=[topfile.split(".")[-1] for topfile in infiles]

# looping over forcefield files
for topfile in infiles:
    print (topfile)
    outfile=topfile.split(".")
    outfile=".".join(outfile[:-1]+["%dnir"%nrep]+outfile[-1:])
    check_repeat=[]
    with open (topfile) as fin:
        fout=open(outfile,"w+")

        if topfile.endswith(".top"):
            # loading and converting .top ff-file
            tag=str()   # .top section header. excample [ bonds ]

            for line in fin:
                if not line.strip().startswith(";"):
                    # remove comments at the end of some lines
                    line=line.rstrip().split(";")[0]+"\n"

                if len(line.strip())==0 or line.strip().startswith(";"):
                    # write lines that are empty or commented 
                    fout.write(line)
                    continue

                if "[" in line:
                    # load the section header/tag
                    tag=line.strip()
                    tag=tag.strip("[]").strip()
                    
                    # check if non-bond params are to be written in .top file
                    # if no xml, writing [ nonbond_params ] section just 
                    # before [ moleculetype ] section 
                    if tag=="moleculetype" and "xml" not in filetypes:
                        fout.write("\n[ nonbond_params ]\n")
                        fout.write("; i\tj\tfunc\tCX\tC12\n")
                        # params for atomtypes with same replica-index
                        for x in range(len(atomtypes)):
                            fout.write(2*(" %s "%(atomtypes[x])))
                            fout.write(" 1 ")                            
                            fout.write(2*" %s "%tuple(CX_C12[x]))
                            fout.write("\n")
                        # params for atom types with different replica-index
                        for x in range(len(atomtypes)):
                            for y in range(x+1,len(atomtypes)):
                                fout.write(" %s  %s " %\
(atomtypes[x],atomtypes[y]))
                                fout.write(" 1 ")                            
                                fout.write(2*("0.0".rjust(10)))
                                fout.write("\n")
                        fout.write("\n")

                    # write the tag line and skip rest of loop
                    fout.write(line)
                    continue
                
                # if last identified tag was atomtypes, edit lines 
                if tag=="atomtypes":
                    for x in range(nrep):
                        i=x+1
                        atname=line.strip().split()[0]
                        new_line=line
                        # write each atom type line for each replica
                        # add replica-index (i) to the atom type name
                        new_atname="%s%d"%(atname,i)
                        new_line=str(new_atname).join(new_line.split(atname))
                        # add to the new atom types and nonbond params list
                        atomtypes.append(new_atname)
                        CX_C12.append(line.split()[-2:])
                        # write modified line
                        fout.write(new_line)
                    continue

                # if last identified tag was atoms, edit atoms lines 
                if tag=="atoms":
                    # load atom number from line
                    atnum=int(line.split()[0])
                    atname=line.split()[1]
                    if atnum%natoms_per_rep==1:
                        # Increment rep counter if atom number 
                        # is 1 more than multiple atoms in 1 rep
                        #load mol id as subtag
                        rep_index+=1
                        subtag="%d"%rep_index
                    # edit atom type in the atoms line to add replica-index
                    #new_atname="%s%d"%(atname,rep_index)
                    line=line.split(atname)
                    line[1]=subtag+line[1]
                    line=atname.join(line)
                    fout.write(line)
                    continue
                fout.write(line)
            fout.close()
        
        if topfile.endswith('.xml'):
            # loading and converting .xml ff-file
            in_nonbond_section=False
            in_contacts_section=False
            nb_params=[]
            for line in fin:
                # list of supported section tags in OpenSMOG
                if line.strip() in ('<OpenSMOGforces>','</OpenSMOGforces>'):
                    fout.write(line)
                if line.strip()=='<nonbond>':
                    in_nonbond_section=True
                if line.strip()=='</nonbond>':
                    fout.write(line)
                    in_nonbond_section=False
                if line.strip()=='<contacts>':
                    in_contacts_section=True
                if line.strip()=='</contacts>':
                    fout.write(line)
                    in_contacts_section=False
                if in_nonbond_section:
# modify expression to disable interactions b/w replicas
                    if 'expr' in line:
                        line=line.split('"')
                        # NIR replica index check condition
                        rep_chk=\
'select(rep_chk(type1,type2),V_intra, V_inter); V_inter=0.0; V_intra='
                        line[1]=rep_chk+line[1]
                        line='"'.join(line)
                        # adding line for extra rep_chk parameter
                        line+='   <parameter>rep_chk</parameter>\n'
                    if "<nonbond_param" in line: 
                        nb_params.append(line)
                        continue
                    if '</nonbond_bytype>' in line:
                        for l in nb_params:
                            # writing intra-replica interactions
                            for x in range(nrep):
                                new_l=l.split('"')
                                # adding same rep-index 
                                new_l[1]+='%d'%(x+1)
                                new_l[3]+='%d'%(x+1)
                                # adding rep_chk paramter
                                new_l[-1]=' rep_chk="1"'+new_l[-1]
                                new_l='"'.join(new_l)
                                fout.write(new_l)
                            for x in range(nrep):
                                for y in range(nrep):
                                    # ignore if same rep-index 
                                    if x==y: continue
                                    new_l=l.split('"')
                                    # if same atom name, only add once
                                    if new_l[1]==new_l[3]:
                                        if x>y: continue
                                    # adding same rep-index 
                                    new_l[1]+='%d'%(x+1)
                                    new_l[3]+='%d'%(y+1)
                                    # adding rep_chk parameter
                                    new_l[-1]=' rep_chk="0"'+new_l[-1]
                                    new_l='"'.join(new_l)
                                    fout.write(new_l)
                    fout.write(line)
                    continue
                if in_contacts_section:
                    # continue writing rest of the lines
                    fout.write(line)

fout.close()


