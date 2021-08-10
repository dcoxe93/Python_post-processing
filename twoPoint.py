from mpi4py import MPI
from mpi4py_fft import PFFT
from mpi4py_fft import newDistArray,DistArray
from mpi4py_fft import HDF5File
from mpi4py_fft.pencil import Subcomm

import pickle
import numpy as np
from scipy.special import roots_legendre
from filereadpar import parfread

import time
import sys

import matplotlib.pyplot as plt
import h5py

comm = MPI.COMM_WORLD
me = comm.Get_rank()
size = comm.Get_size()
expir = 'STD'
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

count = 1
if expir == 'STD':
    skip = 32
    st = 0
else:
    skip = 32
    st = 7

pklT = 'RE{0}/stresses/RE{0}_{1}_{2}.pkl'

nby = 2
nbz = size // nby
R = 0.5
ypl = 30
rpl = (Reynolds - ypl)*R/Reynolds
# r = R*(x+1)/2
xpl = 2*rpl/R - 1
init=True
for ph in range(st,nph,8):
    count = 1

    pklN = pklT.format(Reynolds,expir,ph)
    with open(pklN,'rb') as f:
        stresses = pickle.load(f)
    uvw_b = stresses[0][0:3]

    for i in range(ph,nfiles,skip):
        tind = time.time()
        out = parfread(comm,files[i].strip(),me,size,nby)
        e = time.time() - tind
        tind = time.time()
        if me == 0:
            print('Time to load file # {0}: {1} {2}'.format(i,e,files[i].strip()),flush=True)
        tind = time.time()
        if init:
            ind = 0

            nz, ny, nx = out[0].shape
            ngz = nz*nbz
            ngy = ny*nby
            ngx = nx

            sy = (me % nby)*ny
            ey = sy + ny

            sz = (me // nby)*nz
            ez = sz + nz
            nflds = 3
            t = np.zeros((nz,ny,nx))
            t[:] = np.linspace(0,2*np.pi,nx,endpoint=False)
            # create geometry array
            szXYZ = np.array((nflds,nz*nbz,ny*nby,nx),dtype=int)
            xyz = PFFT(comm,shape=szXYZ,axes=(1,3),grid=(1,nbz,nby,1),ndtype=np.double)

            # create global outerproduct array constructor
            szOP = np.array((nflds,nflds,nz*nbz,ny*nby,nx),dtype=int)
            OPc = PFFT(comm,shape=szOP,axes=(2,4),grid=(1,1,nbz,nby,1),dtype=np.double)
            init = False

        vr = out[0]*np.cos(t) + out[1]*np.sin(t)
        vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
        vz = out[2]

        out[0][:,:,:] = vz[:,:,:]-uvw_b[0][sy:ey].reshape(1,ny,1)
        out[1][:,:,:] = vr[:,:,:]-uvw_b[1][sy:ey].reshape(1,ny,1)
        out[2][:,:,:] = vt[:,:,:]-uvw_b[2][sy:ey].reshape(1,ny,1)

        plt.show()


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
 
        fldsT[:] = xyz.forward(flds)
 

        # redistribute to be contiguous along radial direction
        fldsTr=fldsT.redistribute(2)

        # calculate interpolation parameters
        nf1,lkz,lky,lkx = fldsTr.shape
        if count == 1:
            xi,w = roots_legendre(lky)
            ki = np.linspace(1,lky,lky)
            lam_k = ((-1)**ki)*np.sqrt( (1-xi**2)*w/2)
            dx = xpl - xi
            w_y = lam_k/dx
            s_w_y = np.sum(w_y)

        # outerproduct and average
        beta = 1/count
        alpha = 1-beta
        # 
        fldsr = np.dot(fldsTr.swapaxes(2,3),w_y)/s_w_y
        fldsr = fldsr[:,:,np.newaxis,:]
        for ifld1 in range(nflds):
            for ifld2 in range(nflds):
                fldsOPr[ifld1,ifld2] =( 
                        fldsOPr[ifld1,ifld2]*alpha + 
                        beta*fldsTr[ifld2]*np.conjugate(fldsr[ifld1])
                        )/(ngx*ngz)
                '''
                for kz in range(lkz):
                    for kx in range(lkx):
                        fldsOPr[ifld1,ifld2,kz,:,kx]=alpha*fldsOPr[ifld2,ifld1,kz,:,kx] + beta*(
                                fldsTr[ifld2,kz,:,kx]*
                                np.conj(np.sum(fldsTr[ifld1,kz,:,kx]*w_y))/s_w_y)*(ngx*ngz)
                '''
        count = count + 1
        if me == 0: print('{0}: Time to transform: {1}'.format(count,time.time()-tind),flush=True)

    del flds, fldsT,fldsr
    Rxy = newDistArray(OPc,False)
    fldsOP[:] = fldsOPr.redistribute(2)
    fldsOP = fldsOP
    Rxy = OPc.backward(fldsOP,Rxy)


    del fldsOP, fldsOPr

    fname = 'RE{0}/Rxy_RE{0}_{1}_{2}_yp{3}.h5'.format(Reynolds,expir,ph,int(ypl))
    Rxy.write(fname,'corr',step=0)
