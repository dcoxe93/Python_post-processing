from mpi4py import MPI
from mpi4py_fft import PFFT
from mpi4py_fft import newDistArray,DistArray
from mpi4py_fft import HDF5File
from mpi4py_fft.pencil import Subcomm

import numpy as np
from scipy.special import roots_legendre
from filereadpar import parfread

import time
import sys

comm = MPI.COMM_WORLD
me = comm.Get_rank()
size = comm.Get_size()
Reynolds = 170
for expir in ['STD','OSC']:
    if (size % 2) != 0 and size != 1:
        if me == 0:
            print('# of MPI ranks must be even at this time', file=sys.stderr)
        comm.barrier()
        sys.exit(1)

    if me == 0:
        if expir == 'STD':
            print('loading file list',flush=True)
            nph = 1
            f = open('RE{1}/file_{0}_{1}_1'.format(expir,Reynolds))
            files = f.readlines()
            nfiles = len(files)
        else:
            nph = 32
            nfiles=3200
            files = ['']*nfiles
            for j in range(32):
                for i in range(100):
                        f = open('RE{1}/file_{0}_{1}_{2}'.format(expir,Reynolds,j+1))
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
    init = True

    count = 1
    if expir == 'STD':
        skip = 1
        nph = 1
    else:
        skip = 1
        nph = 1

    nby = 4
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
                print(files[i].strip(),flush=True)
            tind = time.time()
            if init:
                ind = 0
                [nz, ny, nx] = out[0].shape
                ngz = nz*nbz
                ngx = nx
                ngy = nby*ny
                nflds = 3
                t = np.linspace(0,nx,nx,endpoint=False)*2*np.pi/nx
                # create geometry array
                szXYZ = np.array((nflds,nz*nbz,ny*nby,nx),dtype=int)
                xyz = PFFT(comm,shape=szXYZ,axes=(1,3),grid=(1,nbz,nby,1),ndtype=np.double)

                # create global outerproduct array constructor
                szOP = np.array((nflds,nflds,ny*nby,nz*nbz,ny*nby,nx),dtype=int)
                OPc = PFFT(comm,shape=szOP,axes=(3,5),grid=(1,1,1,nbz,nby,1),dtype=np.double)

            vr = out[0]*np.cos(t) + out[1]*np.sin(t)
            vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
            vz = out[2]

            out[0][:,:,:] = vz[:,:,:]
            out[1][:,:,:] = vr[:,:,:]
            out[2][:,:,:] = vt[:,:,:]

            del vr,vt,vz
            
            if init:
                if me == 0: print('initializing distributed arrays',flush=True)
                # contstruct distributed field arrays
                fldsOP = newDistArray(OPc,True) # array for storing outerproduct
                fldsOPr = fldsOP.redistribute(4)
                del fldsOP
                flds = newDistArray(xyz,False) # raw field data
                fldsT = newDistArray(xyz,True)
                init = False

            # transform u,v,w
            for ifld in range(nflds):
                flds[ifld,:,:,:] = out[ifld]
     
            fldsT[:] = xyz.forward(flds,normalize=True)
     

            # redistribute to be contiguous along radial direction
            fldsTr=fldsT.redistribute(2)

            # calculate interpolation parameters
            nf1,lkz,lky,lkx = fldsTr.shape

            # outerproduct and average
            beta = 1/count
            alpha = 1-beta
            # 
            ffld = 0 
            for ifld1 in range(nflds):
                for ifld2 in range(nflds):
                        for kz in range(lkz):
                            for kx in range(lkx):
                                E1 = fldsTr[ifld1,kz,:,kx]
                                E2 = np.conj(fldsTr[ifld2,kz,:,kx])
                                R = fldsOPr[ifld1,ifld2,:,kz,:,kx]
                                fldsOPr[ifld1,ifld2,:,kz,:,kx] = alpha*R+beta*E1[np.newaxis,:]*E2[:,np.newaxis]
                        ffld+=1
            count = count + 1
            if me == 0: print('{0}: Time to transform: {1}'.format(count,time.time()-tind),flush=True)

        del flds, fldsT

        fname = 'RE{0}/Exy_RE{0}_{1}_full.h5'.format(Reynolds,expir)
        fldsOPr.write(fname,'spec',step=ph)
        comm.Barrier()
        for icpu in range(size):
            if me == icpu:
                print(me,'done writing data',flush=True)
            comm.Barrier()
        del fldsOPr,fldsT,fldsTr,fldsT
