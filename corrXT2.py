from mpi4py import MPI
from mpi4py_fft import PFFT
from mpi4py_fft import newDistArray,DistArray
from mpi4py_fft import HDF5File
from mpi4py_fft.pencil import Subcomm

import numpy as np
from filereadpar1 import parfread

import time
import sys

comm = MPI.COMM_WORLD
me = comm.Get_rank()
size = comm.Get_size()
expir = 'OSC'
Reynolds = 360
if (size % 2) != 0 and size != 1:
    if me == 0:
        print('# of MPI ranks must be even at this time', file=sys.stderr)
    comm.barrier()
    sys.exit(1)

if me == 0:
    if expir == 'STD':
        print('loading file list',flush=True)
        nph = 1
        f = open('file_{0}_{1}_1'.format(expir,Reynolds))
        files = f.readlines()
        nfiles = len(files)
    else:
        nph = 32
        nfiles=3200
        files = ['']*nfiles
        for j in range(32):
            for i in range(100):
                    f = open('file_{0}_{1}_{2}'.format(expir,Reynolds,j+1))
                    fl = f.readlines()
                    ng = j + i*32
                    files[ng] = fl[i]

else:
    files = None
    nfiles = None
    nph = None

files = comm.bcast(files,root=0)
nfiles = comm.bcast(nfiles,root=0)
nph = comm.bcast(nph,root=0)
nfiles = 3200//1

count = 1
if expir == 'STD':
    skip = 1
else:
    skip = 32

nby = 2
nbz = size // nby
for ph in range(nph):
    count = 1
    for i in range(ph,nfiles,skip):
        tind = time.time()
        out = parfread(comm,files[i].strip(),me,size,nby)
        e = time.time() - tind
        tind = time.time()
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
            # create geometry array
            szXYZ = np.array((nflds,nz*nbz,ny*nby,nx),dtype=int)
            xyz = PFFT(comm,shape=szXYZ,axes=(1,3),grid=(1,nbz,nby,1),ndtype=np.double)

            # create global outerproduct array constructor
            szOP = np.array((nflds,nflds,nz*nbz,ny*nby,nx),dtype=int)
            OPc = PFFT(comm,shape=szOP,axes=(2,4),grid=(1,1,nbz,nby,1),dtype=np.double)

        t = np.arctan2(y,x)
        vr = out[0]*np.cos(t) + out[1]*np.sin(t)
        vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
        vz = out[2]

        out[0][:,:,:] = vz[:,:,:]
        out[1][:,:,:] = vr[:,:,:]
        out[2][:,:,:] = vt[:,:,:]

        del vr,vt,vz
        
        if count == 1:
            if me == 0: print('initializing distributed arrays',flush=True)
            # contstruct distributed field arrays
            flds = newDistArray(xyz,False) # raw field data
            fldsT = newDistArray(xyz,True)
            fldsOP = newDistArray(OPc,True) # array for storing outerproduct
            fldsOPr = fldsOP.redistribute(3)

        # transform u,v,w
        for ifld in range(nflds):
            flds[ifld,:,:,:] = out[ifld]
 
        fldsT[:] = xyz.forward(flds,normalize=False)
 
        fldsTr=fldsT.redistribute(2)

        # outerproduct and average
        beta = 1/count
        alpha = 1-beta
        for ifld1 in range(nflds):
            for ifld2 in range(nflds):
                fldsOPr[ifld2,ifld1,:,:,:] = alpha*fldsOPr[ifld2,ifld1,:,:,:] + beta*fldsTr[ifld2,:,:,:]*np.conj(fldsTr[ifld1,:,:,:])
        count = count + 1
        if me == 0: print('{0}: Time to transform: {1}'.format(count,time.time()-tind),flush=True)

    del flds, fldsT
    Rxy = newDistArray(OPc,False)
    fldsOP = fldsOPr.redistribute(2)
    Rxy = OPc.backward(fldsOP,Rxy)
    del fldsOP, fldsOPr

    Rxy.write('Rxy_RE{0}_{1}_{2}.h5'.format(Reynolds,expir,ph),'corr',step=0)
    comm.Barrier()
    for icpu in range(size):
        if me == icpu:
            print(me,'done writing data',flush=True)
        comm.Barrier()
    if ph == 0:
        ifxyo = True
    else:
        ifxyo = False

    if ifxyo == True:
        xyzO = newDistArray(xyz,False)
        xyzO[0,:,:,:] = x
        xyzO[1,:,:,:] = y
        xyzO[2,:,:,:] = z
        xyzO.write('./RE{0}/Rxy_RE{0}_{1}_{2}.h5'.format(Reynolds,expir,ph),'xyz',step=0)

        del xyzO
        comm.Barrier()
        for icpu in range(size):
            if me == icpu:
                print(me,'done writing mesh',flush=True)
            comm.Barrier()
