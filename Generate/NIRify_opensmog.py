import numpy as np
import sys


cbdict={
            "CA":"CA",\
            "CBA":"CBA" ,\
            "CBC":"CBC" ,\
            "CBD":"CBD",\
            "CBE":"CBE",\
            "CBF":"CBF" ,\
            "CBG":"CBG" ,\
            "CBH":"CBH",\
            "CBI":"CBI" ,\
            "CBK":"CBK",\
            "CBL":"CBL" ,\
            "CBM":"CBM" ,\
            "CBN":"CBN" ,\
            "CBP":"CBP" ,\
            "CBQ":"CBQ" ,\
            "CBR":"CBR",\
            "CBS":"CBS",\
            "CBT":"CBT",\
            "CBV":"CBV",\
            "CBW":"CBW",\
            "CBY":"CBY"\
        }
del (sys.argv[0])
infiles=[x for x in sys.argv if x.endswith((".top",".xml"))]
nmol=int([x for x in sys.argv if not x.endswith((".top",".xml"))][0])
mol_count=0
atomtypes=[]
for topfile in infiles:
    print (topfile)
    outfile=topfile.split(".")
    outfile=".".join(outfile[:-1]+["ghost%03dnir"%nmol]+outfile[-1:])
    check_repeat=[]
    with open (topfile) as fin:
        fout=open(outfile,"w+")
        if topfile.endswith(".top"):
            tag=str()
            for line in fin:
                if not line.strip().startswith(";"):
                    #remove comments at the end of some lines
                    line=line.rstrip().split(";")[0]+"\n"
                if "[" in line:
                    #load the section header/tag
                    tag=line.strip()
                    tag=tag.strip("[]").strip()
                if tag=="atoms":
                    if line.strip().startswith(";"):
                        if "prot" in line or "nucl" in line:
                            #load mol id as subtag
                            mol_count+=1
                            subtag="%03d"%mol_count
                            #subtag="%03d"%int(line.split("_")[-1])
                            #assert int(subtag)==mol_count
                    if "CA" in line:
                        #add subtag for CA
                        line=line.split("CA")
                        line[1]=subtag+line[1]
                        line="CA".join(line)
                    if "CB" in line:
                        #add subtag for CA
                        line=line.split("CB")
                        line[1]=subtag+line[1]
                        line="CB".join(line)
                    fout.write(line)
                    continue
                if tag=="atomtypes" and len(line.strip())>0 and line.strip()[0] not in "[;":
                    for x in range(nmol):
                        i=x+1
                        new_line=line
                        if "CA" in line:
                            new_line=str("CA%03d"%i).join(new_line.split("CA"))
                            atomtypes.append("CA%03d"%i)
                        if "CB" in line:
                            restype=new_line.split("CB")[1][0]
                            assert "CB"+restype in cbdict
                            new_line=str("CB%03d"%i).join(new_line.split("CB"))
                            atomtypes.append("CB%03d%s"%(i,restype))
                        fout.write(new_line)
                    continue
                fout.write(line)
            fout.close()
            for x in atomtypes:
                print (x)
        if topfile.endswith(".xml"):
            nobond=False
            for line in fin:
                if "<nonbond_bytype>" in line: nobond=True
                if nobond and 'expr' in line:
                    line=line.split('"')
                    extra='select(C12(type1,type2),Vnb,Vghost);Vghost=0.0;Vnb='
                    line[1]=extra+line[1]
                    line='"'.join(line)
                    print (line)
                if "nonbond_param" in line:
                    for x in range(nmol):
                        i=x+1
                        new_line=line
                        if "CA" in line:
                            new_line=str("CA%03d"%i).join(new_line.split("CA"))
                            params=line
                        if "CB" in line:
                            new_line=str("CB%03d"%i).join(new_line.split("CB"))
                        fout.write(new_line)
                    continue
                if "/nonbond_bytype" in line:
                    atomtypes=list(set(atomtypes))
                    atomtypes.sort()
                    f1=params.split('=')[0]
                    params=[x.split('=') for x in params.split()[1:]]
                    params[0][0]=f1
                    for x in range(2,len(params)-1): params[x][1]='"0"'
                    for x in range(0,len(atomtypes)):
                        for y in range(x+1,len(atomtypes)):
                            params[0][1],params[1][1]='"%s"'%atomtypes[x],'"%s"'%atomtypes[y]
                            new_line=['='.join(x) for x in params]
                            new_line=" ".join(new_line)+"\n"
                            fout.write(new_line)
                    nobond=False
                fout.write(line)
