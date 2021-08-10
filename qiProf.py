import numpy as np
from mpi4py import MPI
from filereadpar1 import parfread

import os.path
from os import path
import time
import sys
import pickle

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
        nph = 1
        f = open('./RE{1}/file_{0}_{1}_1'.format(expir,Reynolds))
        files = f.readlines()
        nfiles = len(files)
    else:
        nph = 32
        files = ['']*3200
        for j in range(32):
            for i in range(100):
                    f = open('./RE{1}/file_{0}_{1}_{2}'.format(expir,Reynolds,j+1))
                    fl = f.readlines()
                    ng = j + i*32
                    files[ng] = fl[i]

else:
    files = None
    nph = None

files = comm.bcast(files,root=0)
nph = comm.bcast(nph,root=0)
nfiles = int(len(files)/1)

count = 1
if expir == 'STD':
    skip = 4
else:
    skip = 32
nby = 2
for ph in range(nph):
    # load the mean fields
    savename = './RE{0}/mflds_{1}_{2}.pkl'.format(Reynolds,expir,ph)
    count = 1
    if path.exists(savename) == False:
        for i in range(ph,nfiles,skip):
            tind = time.time()
            out = parfread(comm,files[i].strip(),me,size,nby)
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
            else:
                vx = out[0]
                vy = out[1]
                vz = out[2]

            t = np.arctan2(y,x)
            r = np.sqrt(x**2 + y**2)
            vr = out[0]*np.cos(t) + out[1]*np.sin(t)
            vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
            vz = out[2]

            out[0][:,:,:] = vz[:,:,:]
            out[1][:,:,:] = vr[:,:,:]
            out[2][:,:,:] = vt[:,:,:]

            if count == 1:
                stats = np.zeros((nflds,ny),dtype=np.double)
            beta = 1/count
            alpha = 1 - beta

            stats[0,:] = stats[0,:]*alpha + beta*np.mean(out[0],axis=(0,2))
            stats[1,:] = stats[1,:]*alpha + beta*np.mean(out[1],axis=(0,2))
            stats[2,:] = stats[2,:]*alpha + beta*np.mean(out[2],axis=(0,2))
            count+=1
                    
        statsT0 = np.zeros((nflds,ny*nby),dtype=np.double)
        statsT1 = statsT0.copy()
        iy = (me % nby)*ny
        ey = iy + ny
        statsT0[:,iy:ey] = stats

        comm.Reduce([statsT0,MPI.DOUBLE],[statsT1,MPI.DOUBLE])
        statsT1 = statsT1/(size/nby)
        if me == 0:
            print(ph,'writing data: {0}'.format(savename),flush=True)
            with open(savename,'wb') as f:
                pickle.dump([statsT1],f,protocol=-1)
        comm.barrier()
    else:
        if me == 0:
            print(ph,'opening: {0}'.format(savename),flush=True)
            with open(savename,'rb') as f:
                statsT = pickle.load(f)
                statsT = statsT[0]
        else:
            statsT = None

        statsT= comm.bcast(statsT,root=0)
        nflds,ngy = statsT.shape
        ny = ngy // nby 
        iy = (me % nby)*ny
        ey = iy+ny
        
        stats = statsT[:,iy:ey] 

    # begin calculating the quadrant analysis
    statsQi = np.zeros((nflds,4,ny),dtype = np.double)
    nQi = np.zeros((nflds,4,ny),dtype = np.double)
    for i in range(ph,nfiles,skip):
        tind = time.time()
        out = parfread(comm,files[i].strip(),me,size,nby)
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
        else:
            vx = out[0]
            vy = out[1]
            vz = out[2]

        t = np.arctan2(y,x)
        r = np.sqrt(x**2 + y**2)
        vr = out[0]*np.cos(t) + out[1]*np.sin(t)
        vt =-out[0]*np.sin(t) + out[1]*np.cos(t)
        vz = out[2]

        out[0][:,:,:] = vz[:,:,:]
        out[1][:,:,:] = vr[:,:,:]
        out[2][:,:,:] = vt[:,:,:]

        # subtract the long time/phase average mean from the velocity field
        for ifld in range(nflds):
            out[ifld][:,:,:] = out[ifld][:,:,:] - stats[ifld,np.newaxis,:,np.newaxis]
        
        # loop over the velocity fields calculating the mean contribution to eache velocity pair
        for ifld1 in range(nflds):

            # choose second component
            if ifld1 == 0:
                # if streamwise then calculate < u'v' >_qi
                ifld2 = 1
            elif ifld1 == 1:
                # if radial then calculate < v'w' >_qi
                ifld2 = 2
            else:
                # if azimuthal then calculate < w'u' >_qi
                ifld2 = 0
            mm1 = (out[ifld1] >= 0).astype(np.double)
            mm2 = (out[ifld2] >= 0).astype(np.double)
                    
            for m2 in range(2):
                for m1 in range(2):
                    # loop over masks assuming ifld1 is the x-axis
                    md1 = (1-mm1)*m1 + mm1*(1-m1)
                    md2 = (1-mm2)*m2 + mm2*(1-m2)
                    # q = 0,1,2,3 
                    # consider u' v' 
                    # m1 = 0, m2 = 0; q = 0 -> u' > 0, v' < 0: Q_4
                    # m1 = 1, m2 = 0; q = 1 -> u' < 0, v' < 0: Q_3
                    # m1 = 0, m2 = 1; q = 2 -> u' > 0, v' > 0: Q_1
                    # m1 = 0, m2 = 1; q = 3 -> u' < 0, v' > 0: Q_2
                    q = m1 + 2*m2

                    nQi[ifld1,q,:] += (
                            np.sum(md1*md2,axis=(0,2))
                            )
                    statsQi[ifld1,q,:] += (
                            np.sum(md1*md2*out[ifld1]*out[ifld2],axis=(0,2))
                            )

    statsQiG0 = np.zeros((nflds,4,ny*nby),dtype=np.double)
    statsQiG1 = statsQiG0.copy()

    nQiG0 = np.zeros((nflds,4,ny*nby),dtype=np.double)
    nQiG1 = nQiG0.copy()

    iy = (me % nby)*ny
    ey = iy + ny
    statsQiG0[:,:,iy:ey] = statsQi
    nQiG0[:,:,iy:ey] = nQi

    comm.Reduce([statsQiG0,MPI.DOUBLE],[statsQiG1,MPI.DOUBLE])
    comm.Reduce([nQiG0,MPI.DOUBLE],[nQiG1,MPI.DOUBLE])

    savename = './RE{0}/statsQi_{1}_{2}.pkl'.format(Reynolds,expir,ph)
    if me == 0:
        print(ph,'writing quadrants: {0}'.format(savename),flush=True)
        with open(savename,'wb') as f:
            pickle.dump([statsQiG1,nQiG1],f,protocol=-1)
