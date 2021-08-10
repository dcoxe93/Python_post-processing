from mpi4py import MPI
from filereadpar1 import parfread
import time
import numpy as np
from scipy.fft import rfft, irfft
import matplotlib.pyplot as plt
import h5py
import sys

comm = MPI.COMM_WORLD
me = comm.Get_rank()
size = comm.Get_size()
print('',flush=True)
expir = 'OSC'
Reynolds = 170
if (size % 2) != 0 and size != 1:
    if me == 0:
        print('# of MPI ranks must be even at this time', file=sys.stderr)
    comm.barrier()
    sys.exit(1)

if me == 0:
    if expir == 'STD':
        f = open('file_{0}_{1}_1'.format(expir,Reynolds))
        files = f.readlines()
        nfiles = len(files)
    else:
        files = ['']*3200
        for j in range(32):
            for i in range(100):
                    f = open('file_{0}_{1}_{2}'.format(expir,Reynolds,j+1))
                    fl = f.readlines()
                    ng = j + i*32
                    files[ng] = fl[i]

else:
    files = None
files = comm.bcast(files,root=0)
nfiles = len(files)//1
nfpf = 1600
jump = 128
count = 1
si = 0
ei = nfpf
ti = 0
W = np.sin(np.pi*np.linspace(0,nfpf-1,nfpf)/nfpf)
W = W*W
Wm= np.sum(W)
nsubx = 4
nsubz = 8
nby = 1
while ei <= nfiles:
    for i in range(si,ei):
        tind = time.time()
        out = parfread(comm,files[i].strip(),me,size,nby)
        e = time.time() - tind
        if me == 0:
            print('Time to load file # {0}: {1}'.format(i,e),flush=True)
        tind = time.time()
        if i == 0:
            ind = 0
            vx = out[0]
            vy = out[1]
            vz = out[2]
            x = out[3]
            y = out[4]
            z = out[5]
            [nz, ny, nx] = vx.shape
            nsz = nz // nsubz
            nsx = nx // nsubx
            nflds = 3
            tseries = np.zeros((nfpf,nflds,nsz,ny,nsx),dtype=np.double)
        else:
            vx = out[0]
            vy = out[1]
            vz = out[2]

        t = np.arctan2(y,x)
        vr = vx*np.cos(t) + vy*np.sin(t)
        vt =-vx*np.sin(t) + vy*np.cos(t)

        tseries[ind,0,:,:,:] =vz[::nsubz,:,::nsubx].reshape(1,1,int(nz/nsubz),ny,int(nx/nsubx))
        tseries[ind,1,:,:,:] =vr[::nsubz,:,::nsubx].reshape(1,1,int(nz/nsubz),ny,int(nx/nsubx))
        tseries[ind,2,:,:,:] =vt[::nsubz,:,::nsubx].reshape(1,1,int(nz/nsubz),ny,int(nx/nsubx))
        ind = ind + 1
    tmean = np.zeros((nflds,nsz,ny,nsx),dtype=np.double)
    ts1 = np.zeros((nfpf//2 +1,nflds,nsz,ny,nsx),dtype=np.cdouble)
    for n in range(nflds):
        for k in range(nsz):
            for j in range(ny):
                for i in range(nsx):
                   tmean[n,k,j,i] = np.sum(tseries[:,n,k,j,i]*W)/Wm
                   ts1[:,n,k,j,i] = rfft((tseries[:,n,k,j,i]-tmean[n,k,j,i])*W)
    if count == 1:
        TS = np.empty((nflds,nflds,nfpf//2 + 1,ny),dtype=np.cdouble)
        for j in range(nflds):
            for i in range(nflds):
                TS[j,i,:,:] = np.squeeze(np.mean(ts1[:,j,:,:,:]*np.conj(ts1[:,i,:,:,:]),axis=(1,3)))
    else:
        beta = 1./count
        alpha = 1.-beta
        for j in range(nflds):
            for i in range(nflds):
                TS[j,i,:,:] =TS[j,i,:,:]*alpha + beta*np.squeeze(np.mean(ts1[:,j,:,:,:]*np.conj(ts1[:,i,:,:,:]),axis=(1,3)))

    count = count + 1
    si = ei
    ei = ei + jump
    ind = nfpf-jump
    if ei <= nfiles:
        tseries[0:(nfpf-jump),:,:,:,:]=tseries[jump:,:,:,:,:]
    e= time.time()-tind
    if me == 0: print('Time to transform: {0}'.format(e),flush=True)
    
del tseries,z

if size > 1:
    TSg0 = np.zeros((nflds,nflds,nfpf//2 + 1,ny*nby),dtype=TS.dtype)
    TSg1 = TSg0.copy()
    iy = (me % nby)*ny
    ey = iy + ny
    TSg0[:,:,:,iy:ey] = TS
    del TS
    comm.Reduce([TSg0, MPI.COMPLEX],[TSg1, MPI.COMPLEX])
    TSg1 = TSg1/(size / nby)
    xv = x[0,:,0]
    yv = y[0,:,0]
    r = np.squeeze(np.sqrt(xv**2 + yv**2))
    rG0 = np.zeros((ny*nby,),dtype=np.double)
    rG1 = rG0.copy()
    rG0[iy:ey] = r
    comm.Reduce([rG0,MPI.DOUBLE],[rG1,MPI.DOUBLE])
    rG1 = rG1 / (size/nby)
    if me == 0:
        xv = x[0,:,0]
        yv = y[0,:,0]
        r = np.squeeze(np.sqrt(xv*xv + yv*yv))
    
        with h5py.File('RE{0}_{1}.h5'.format(Reynolds,expir),'w') as hf:
            TSdat = hf.create_dataset('spectra',TSg1.shape,dtype=TSg1.dtype,data=TSg1)
            TSdat[:] = TSg1
            rdat = hf.create_dataset('r',rG1.shape,dtype=rG1.dtype,data=rG1)
            rdat[:] = rG1
else:
    flds = ['uu','uv','uw','vu','vv','vw','wu','wv','ww']
    xv = x[1,:,1]
    yv = y[1,:,1]
    r = np.squeeze(np.sqrt(xv*xv + yv*yv))
    
    #hf = h5py.File('RE{0}_{1}_par.h5'.format(Reynolds,count),'w',driver='mpio',comm=comm)
    hf = h5py.File('RE{0}_{1}.h5'.format(Reynolds,expir),'w')
    TSdat = hf.create_dataset('spectra',(nflds,nflds,nfpf,ny),dtype=TS.dtype,data=TS)
    rdat = hf.create_dataset('r',r.shape,dtype=r.dtype,data=r)
    hf.close()

