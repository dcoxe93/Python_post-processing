import time
import sys

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import roots_legendre,roots_jacobi,eval_jacobi

from mpi4py import MPI
from mpi4py_fft import PFFT,newDistArray,HDF5File

from filereadpar import parfread



def legDiffM(N):
    D = np.zeros((N,N),dtype=np.double)
    z = np.zeros(N)
    z[0]  =-1
    z[-1] = 1
    z[1:-1],w = roots_jacobi(N-2,1.,1.)
    L1 = eval_jacobi(N-1,0.,0.,z)
    qq1 = N*(N-1)/4
    D[0,0]   =-qq1
    D[-1,-1] = qq1

    for i in range(N):
        for j in range(N):
            if i != j:
                D[i,j] = (L1[i]/L1[j])*1/(z[i] - z[j])
    return D


pi = np.pi
icomm = MPI.COMM_WORLD
me = icomm.Get_rank()
size = icomm.Get_size()
expir = 'STD'
Reynolds = 170
if (size % 2) != 0 and size != 1:
    if me == 0:
        print('# of MPI ranks must be even at this time', file=sys.stderr)
    icomm.barrier()
    sys.exit(1)

if me == 0:
    if expir == 'STD':
        print('loading file list',flush=True)
        nph = 1
        f = open('RE{0}/file_{1}_{0}_1'.format(Reynolds,expir))
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

files = icomm.bcast(files,root=0)
nph = icomm.bcast(nph,root=0)
nfiles = 3200//1

count = 1
if expir == 'STD':
    skip = 32
else:
    skip = 128

nby = 2
nbz = size // nby
R = 0.5
ypl = 50
rpl = (Reynolds - ypl)*R/Reynolds
# r = R*(x+1)/2
xpl = 2*rpl/R - 1
for ph in range(nph):
    count = 1
    for i in range(ph,nfiles,skip):
        tind = time.time()
        out = parfread(icomm,files[i].strip(),me,size,nby)
        e = time.time() - tind
        tind = time.time()
        if me == 0:
            print('Time to load file # {0}: {1}'.format(i,e),flush=True)
        tind = time.time()
        if i == 0:
            ind = 0
            [nz, ny, nx] = out[0].shape
            t = np.zeros((nz,ny,nx))
            ngz = nz*nbz
            sz = (me // nby)*nz
            ez = sz + nz

            ngy = ny*nby
            sy = (me % nby)*ny
            ey = sy + ny

            ngx = nx

            tv = np.linspace(0,2*pi,nx,endpoint=False)
            t[:] = tv
            nflds = 3
             # create the FFT arrays
            szXYZ = np.array((ngz,ngy,ngx),dtype=int)
            dxz = PFFT(icomm,shape=szXYZ,axes=(0,2),
                    grid=(nbz,nby,1),dtype=np.cdouble)

            # Initialize the FFT arrays
            u = newDistArray(dxz,False)
            v = newDistArray(dxz,False)
            w = newDistArray(dxz,False)

            kx = newDistArray(dxz,False)
            kz = newDistArray(dxz,False)
            #rv = newDistArray(dxz,False)

            u_hat0 = newDistArray(dxz,True)
            v_hat0 = newDistArray(dxz,True)
            w_hat0 = newDistArray(dxz,True)

            kzv = np.linspace(0,ngz,ngz,endpoint=False)
            kxv = np.linspace(0,nx,nx,endpoint=False)

            kzv = kzv*(kzv < (ngz//2)) + (kzv - ngz)*(kzv >= (ngz // 2))
            kxv = kxv*(kxv < (ngx//2)) + (kxv - ngx)*(kxv >= (ngx // 2))
            x,wi = roots_legendre(ngy)
            r = 0.5*(x+1)/2
            rl = r[sy:ey]
            # populate the wave numbers
            for iz in range(nz):
                izg = iz + ( me // nby)*nz
                for ix in range(nx):
                    kx[iz,:,ix] = kxv[ix]
                    kz[iz,:,ix] = kzv[izg]*(2*np.pi)/12
                    #rv[iz,:,ix] = rl
                    #if izg == ngz//2:kz[iz,:,ix] = 0.
                    #if ix == nx//2:kx[iz,:,ix] = 0.

            gradU = np.empty((3,nz,ny,nx))
            gradV = gradU.copy()
            gradW = gradU.copy()

            if(me == 0): print('initializing vorticity arrays',flush=True)
            omega_r = np.zeros((nz,ny,nx))
            omega_t = np.zeros((nz,ny,nx))
            omega_z = np.zeros((nz,ny,nx))

            omega = np.zeros((9,ngy))

        gradU[:] = 0
        gradV[:] = 0
        gradW[:] = 0

        vr = out[0]*np.cos(tv) + out[1]*np.sin(tv)
        vt =-out[0]*np.sin(tv) + out[1]*np.cos(tv)
        vz = out[2]

        u[:] = vz
        v[:] = vr
        w[:] = vt

        del out,vr,vt,vz
        # Since we have already started, lets calculate the radial derivatives 
        D = 4*legDiffM(ngy)

        ur = u.redistribute(1) 
        ur[:] = np.matmul(D,ur)
        gradU[0][:] = np.real(ur.redistribute(2))
        del ur

        vr = v.redistribute(1) 
        vr[:] = np.matmul(D,vr)
        gradV[0][:] = np.real(vr.redistribute(2))
        del vr

        wr = w.redistribute(1) 
        wr[:] = np.matmul(D,wr)
        gradW[0][:] = np.real(wr.redistribute(2))
        del wr

        # Fourier Transform the velocity fields
        u_hat0[:] = dxz.forward(u,normalize=True)
        v_hat0[:] = dxz.forward(v,normalize=True)
        w_hat0[:] = dxz.forward(w,normalize=True)

        # orient the wavenumbers along the proper axis
        kz_z = kz.redistribute(0)
        kx_z = kx.redistribute(0)

        # calculate the streamwise derivatives
        if i == 0:
            du = np.zeros_like(u)
            dv = np.zeros_like(v)
            dw = np.zeros_like(w)

        du = dxz.backward(1j*kz_z*u_hat0,du)
        dv = dxz.backward(1j*kz_z*v_hat0,dv)
        dw = dxz.backward(1j*kz_z*w_hat0,dw)

        gradU[2] = np.real(du)
        gradV[2] = np.real(dv)
        gradW[2] = np.real(dw)

        du[:] = 0
        dv[:] = 0
        dw[:] = 0

        # calculate the azimuthal derivatives
        du = dxz.backward(1j*kx_z*u_hat0,du)
        dv = dxz.backward(1j*kx_z*v_hat0,dv)
        dw = dxz.backward(1j*kx_z*w_hat0,dw)

        gradU[1] = np.real(du)/rl.reshape((1,ny,1))
        gradV[1] = np.real(dv)/rl.reshape((1,ny,1))
        gradW[1] = np.real(dw)/rl.reshape((1,ny,1))

        du[:] = 0
        dv[:] = 0
        dw[:] = 0
        
        # define averaging parameter
        beta  = 1/count
        alpha = 1-beta
        count+=1
    
        # calculate the mean vorticity statistics
        omega[0,sy:ey] = (omega[0,sy:ey]*alpha
                +beta*np.mean(gradU[0]**2,axis=(0,2)))

        omega[1,sy:ey] = (omega[1,sy:ey]*alpha
                +beta*np.mean(gradV[0]**2,axis=(0,2)))

        omega[2,sy:ey] = (omega[2,sy:ey]*alpha
                +beta*np.mean(gradW[0]**2,axis=(0,2)))

        # calculate the RMS vorticity statistics
        omega[3,sy:ey] = (omega[3,sy:ey]*alpha
                +beta*np.mean(gradU[1]**2,axis=(0,2)))
        omega[4,sy:ey] = (omega[4,sy:ey]*alpha
                +beta*np.mean(gradV[1]**2,axis=(0,2)))
        omega[5,sy:ey] = (omega[5,sy:ey]*alpha
                +beta*np.mean(gradW[1]**2,axis=(0,2)))

        # calculate the co-variance statistics
        omega[6,sy:ey] = (omega[6,sy:ey]*alpha
                +beta*np.mean(gradU[2]**2,axis=(0,2)))

        omega[7,sy:ey] = (omega[7,sy:ey]*alpha
                +beta*np.mean(gradV[2]**2,axis=(0,2)))
        
        omega[8,sy:ey] = (omega[8,sy:ey]*alpha
                +beta*np.mean(gradW[2]**2,axis=(0,2)))


    omega_g = omega.copy()
    omega_g[:] = 0
    icomm.Allreduce([omega,MPI.DOUBLE],[omega_g,MPI.DOUBLE])
    omega_g = omega_g/nbz

    if me == 0:
        '''
        omega_g[3] = omega_g[3] - omega_g[0]**2
        omega_g[4] = omega_g[4] - omega_g[1]**2
        omega_g[5] = omega_g[5] - omega_g[2]**2

        omega_g[6] = omega_g[6] - omega_g[0]*omega_g[1]
        omega_g[7] = omega_g[7] - omega_g[1]*omega_g[2]
        omega_g[8] = omega_g[8] - omega_g[2]*omega_g[0]
        '''

        savetemp = 'RE{0}/grad_{1}_{2}.pkl'
        savename = savetemp.format(Reynolds,expir,ph)
        with open(savename,'wb') as f:
            pickle.dump(omega_g,f)
