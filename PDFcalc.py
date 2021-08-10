# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
#!/usr/bin/python

import numpy as np
import scipy as sp
from scipy.special import roots_legendre
from mpi4py import MPI
from filereadpar import parfread

import time
import sys
import pickle

comm = MPI.COMM_WORLD
me = comm.Get_rank()
size = comm.Get_size()
expir = 'STD'
Reynolds = 360
done = False
if (size % 2) != 0 and size != 1:
    if me == 0:
        print('# of MPI ranks must be even at this time', file=sys.stderr)
    comm.barrier()
    sys.exit(1)

if me == 0:
    if expir == 'STD':
        nph = 1
        f = open('RE{1}/file_{0}_{1}_1'.format(expir,Reynolds))
        files = f.readlines()
        nfiles = len(files)
    else:
        nph = 32
        files = ['']*3200
        for j in range(32):
            for i in range(100):
                    f = open('RE{1}/file_{0}_{1}_{2}'.format(expir,Reynolds,j+1))
                    fl = f.readlines()
                    ng = j + i*32
                    files[ng] = fl[i]

else:
    files = None
    nph = None

files = comm.bcast(files,root=0)
nph = comm.bcast(nph,root=0)
nfiles = 3200
n = 64

count = 1
if expir == 'STD':
    skip = 32
else:
    skip = 32

for ph in range(nph):
    count = 1
    if ph == 0:
        if me == 0:
            for ph0 in range(nph):
                fn = 'RE{0}/RE{0}_{1}_{2}.pkl'.format(Reynolds,expir,ph0)
                with open(fn,'rb') as f:
                    data=pickle.load(f)
                    if ph0 == 0:
                        stats = data[0]
                    else:
                        beta = 1/(ph0+1)
                        alpha = 1 - beta
                        stats = stats*alpha + beta*data[0]
        else:
            stats = None
        stats=comm.bcast(stats,root=0)

    fn = 'RE{0}/RE{0}_{1}_{2}.pkl'.format(Reynolds,expir,ph)
    if me == 0:
        with open(fn,'rb') as f:
            statsL = pickle.load(f)
            statsL = statsL[0]
    else:
        statsL = None
    statsL = comm.bcast(statsL,root=0)

    for i in range(ph,nfiles,skip):
        tind = time.time()
        out = parfread(comm,files[i].strip(),me,size,1)
        e = time.time() - tind
        if me == 0:
            print('Time to load file # {0}: {1}'.format(i,e),flush=True)
        tind = time.time()

        if i == 0:
            ind = 0
            x = out[3]
            y = out[4]
            z = out[5]

            [nz, ny, nx] = x.shape
            nflds = 3
            nyp = ny//2 # downsample by a factor of 4
            rloc,w = roots_legendre(nyp)

            # create the PDF data structures
            jPDFs = np.zeros((nflds,nyp,n,n),dtype=np.double)
            
            # create the bounds
            bdds = np.zeros(nflds)
            bdds[0] = 4*np.sqrt(np.max(stats[3]))
            bdds[1] = 4*np.sqrt(np.max(stats[4]))
            bdds[2] = 4*np.sqrt(np.max(stats[5]))

            bdv = np.zeros((nflds,n+1),dtype=np.double)
            bdv[0] = np.linspace(-bdds[0],bdds[0],n+1)
            bdv[1] = np.linspace(-bdds[1],bdds[1],n+1)
            bdv[2] = np.linspace(-bdds[2],bdds[2],n+1)

            # Setup interpolation parameters
            t = np.arctan2(y,x)
            d,w = roots_legendre(ny)
            del x,y,z
            k = np.linspace(start=1,stop=ny,num=ny)
            dl = ((-1)**k)*np.sqrt(((1-d**2))*w/2)
            bdv = comm.bcast(bdv,root=0)        
            dl = comm.bcast(dl,root=0)
            d = comm.bcast(d,root=0)

        vr = out[0]*np.cos(t) + out[1]*np.sin(t)
        vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
        vz = out[2]

        out[0][:,:,:] = vz[:,:,:]
        out[1][:,:,:] = vr[:,:,:]
        out[2][:,:,:] = vt[:,:,:]

        for iyp in range(nyp): # loop over y+ coordinates
            dx = d - rloc[iyp]
            sc = dl/dx
            ssc = np.sum(dl/dx)
            for ifld1 in range(1): # loop over the first field
                dum2 = 0 # initialize the counter of all points tallied in theta-z plane
                if ifld1 == 0:
                    # if streamwise velocity -> calculate jPDF_uv
                    ifld2 = 1
                elif ifld1 == 1:
                    # if radial velocity -> calculate jPDF_vw
                    ifld2 = 2
                elif ifld1 == 2:
                    # if azimuthal velocity -> calculate jPDF_wu
                    ifld2 = 0
                u1m = np.sum(statsL[ifld1]*sc)/ssc
                u2m = np.sum(statsL[ifld2]*sc)/ssc

                for il in range(0,nx,2): # loop over azimuthal coordinate

                    if ifld1 == 0:
                        # if streamwise velocity -> calculate jPDF_uv
                        ifld2 = 1
                    elif ifld1 == 1:
                        # if radial velocity -> calculate jPDF_vw
                        ifld2 = 2
                    elif ifld1 == 2:
                        # if azimuthal velocity -> calculate jPDF_wu
                        ifld2 = 0



                    # interpolate the 1st field at the selected y+ coordinate
                    u1r = out[ifld1][::2,:,il]
                    u1i = np.dot(u1r,sc)/ssc

                    # subtract the long time mean from the field
                    u1i = u1i - u1m

                    # interpolate the 2nd field at the selected y+ coordinate
                    u2r = out[ifld2][::2,:,il]
                    u2i = np.dot(u2r,sc)/ssc
                    
                    # subtract the mean value
                    u2i = u2i - u2m

                    dum1 = 0
                    for iz in range(0,nz//2,1):
                        u1 = u1i[iz]
                        u2 = u2i[iz]
                        for n1 in range(n):
                            u1u = bdv[ifld1,n1+1]
                            u1l = bdv[ifld1,n1]
                            if u1 >= u1l and u1 < u1u:
                                for n2 in range(n):
                                    u2u = bdv[ifld2,n2+1]
                                    u2l = bdv[ifld2,n2]
                                    if u2 >= u2l and u2 < u2u:
                                        jPDFs[ifld1,iyp,n1,n2] += 1
                                        dum1+=1
                                        break
                                break
                    dum2+=dum1
                
            if int(dum2) != nz*nx//4:
                if np.abs(dum2 - nz*nx//4) > 10:
                    print(me,'Not all values counted in fields {0}, {1}. {2} values not counted.  Consider expanding interval'.format(ifld1,ifld2,nz*nx//4-dum2),flush=True)
        e = time.time() - tind
        if me == 0:
            print('Time to Calc PDF # {0}: {1}'.format(i,e),flush=True)
    jPDFsG = np.zeros(jPDFs.shape,dtype=jPDFs.dtype)
    comm.Reduce([jPDFs,MPI.DOUBLE],[jPDFsG,MPI.DOUBLE],op=MPI.SUM,root=0)
    if me == 0:
        bdvL = bdv[:,:n:]
        bdvR = bdv[:,1:n+1:]
        bdds = (bdvL + bdvR)/2
        with open('RE{0}/PDFs_RE{0}_{1}_{2}.pkl'.format(Reynolds,expir,ph),'wb') as f:
            pickle.dump([bdds,jPDFsG],f,protocol=-1)
