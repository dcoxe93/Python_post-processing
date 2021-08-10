from mpi4py import MPI
import numpy as np

# this module uses MPI to read a data file dumped from my Nek5000 interpolation routines
# It must be used in conjunction with MPI, mpi4py and numpy

# input variables:
# comm - MPI communicator passed from main script
# fname - string; filename
# nid - integer; rank of the mpi process
# nranks - integer; total number of mpi-processes

# output variables:
# vx - np.double; x-component of velocity
# vy - np.double; y-component of velocity
# vz - np.double; z-component of velocity

# optional output variables
# x - np.single; x-cooridnate of each gridpoint
# y - np.single; y-cooridnate of each gridpoint
# z - np.single; z-cooridnate of each gridpoint
# t - np.double; other fields dumped from Nek5000
def parfread(comm,fname,nid,nranks,nby):
    nbz = nranks // nby

    # open the file using MPI
    fh = MPI.File.Open(comm,fname,MPI.MODE_RDONLY)
    # header will be read in as unsigned byte -> header is only 132bytes long
    # can verify length by typing in terminal "head -c 132 fname"
    hdr = np.empty((132),dtype=np.ubyte)
    # have each process read the header individually
    # might be an IO bottle neck if many MPI processes but so far low impact on 64 processes
    # if becomes an issue have rank 0 read the header and broadcast

    fh.Read(hdr)
    HDR = "".join([chr(i) for i in hdr])
       # print(HDR[1:]) # full header
    lx1 = int(HDR[4:7]) # lx1 - number of points in x-direction
    ly1 = int(HDR[12:14]) # ly1 - number of point in y-direction
    lz1 = int(HDR[19:22]) # lz1 - number of points in z-direction
    blx = int(HDR[28:30]) # blx - number of blocks in x-direction (usually = 1)
    bly = int(HDR[36:38]) # bly - number of blocks in y-direction
    blz = int(HDR[43:46]) # blz - number of blocks in z-direction
    nflds = int(HDR[53:54]) # nflds - number of fields written to file
    ifmesh = HDR[60] # ifMesh - string T/F: T = file contains mesh data, F = file does not contain mesh data

    # calculate how many points are in the domain
    nblocks = blx*bly*blz # total number of blocks
    nppb = lx1*ly1*lz1 # points per block
    npts = nppb*nblocks

    # calculate if we can distrute the points evenly among the processor
    if (nblocks % nranks) != 0 or nranks > nblocks:
        if nid == 0:
            print('Number of blocks ({0}) is not divible by the  number of ranks ({1})'.format(nblocks,nranks), file=sys.stderr)
        comm.barrier()
        sys.exit(1)

    lblx = blx
    lbly = bly//nby
    lblz = blz//nbz

    # set how many points in each local direction
    localX = lx1*blx
    localY = ly1*lbly
    localZ = lz1*lblz

    # set the total number of point in a local block
    nppbl = localX*localY*localZ

    # Set header offset
    myoff0 = 132

    xbp = 8*3 # number of bytes per (x,y,z) point 

    # check if the mesh is contained in this file
    if ifmesh == 'T':

        # if mesh is present initialize an array of to fit the number of point this processor will read
        xyzl = np.empty((3*lx1*ly1*lz1),dtype=np.double)

        # set the offset based on y-block starting point and z-block starting point
        # We distribute the points based on how many points are in the local block
        # the number of points in the local block are localX*localY*localZ
        myoff1 = myoff0 + ( nid % nby )*lbly*nppb*xbp + ( nid // nby )*bly*lblz*nppb*xbp

        x = np.zeros((localZ,localY,localX),dtype=np.double)
        y = x.copy()
        z = x.copy()

        for lz  in range(lblz):
            for ly in range(lbly):
                # each block contains lx1*ly1*lz1 points
                myoff = myoff1 + ly*nppb*xbp + lz*bly*nppb*xbp
               
                fh.Set_view(myoff,MPI.BYTE)
                fh.Read_all(xyzl)
                

                ins = 0
                ine = ins + lx1

                jns = 0 + ly*ly1
                jne = jns + ly1

                kns = 0 + lz*lz1
                kne = kns + lz1

#                x[kns:kne,jns:jne,ins:ine]=xyzl[0::3].reshape((lz1,ly1,lx1))
#                y[kns:kne,jns:jne,ins:ine]=xyzl[1::3].reshape((lz1,ly1,lx1))
#                z[kns:kne,jns:jne,ins:ine]=xyzl[2::3].reshape((lz1,ly1,lx1))

                x[kns:kne,jns:jne,:]=xyzl[0::3].reshape((lz1,ly1,lx1))
                y[kns:kne,jns:jne,:]=xyzl[1::3].reshape((lz1,ly1,lx1))
                z[kns:kne,jns:jne,:]=xyzl[2::3].reshape((lz1,ly1,lx1))
                xyzl[:] = 0



    # create new variable to define whether or not the mesh was stored
    # if it was then we need to add to the file the offset the number of bytes the ENTIRE mesh occupied in the file
    meshoff = 0
    if ifmesh == 'T':
        meshoff = 1
    fbp = nflds*8
    myoff1 = myoff0 + meshoff*npts*xbp +  ( nid % nby )*lbly*nppb*fbp + ( nid // nby )*bly*lblz*nppb*fbp

   
    uvwl = np.empty(nflds*lx1*ly1*lz1,dtype=np.double)

    # create 3d storage arrays for the points of velocity data and scalar field data
    vx = np.zeros((localZ,localY,localX),dtype=np.double)
    vy = vx.copy()
    vz = vx.copy()

    t = np.zeros((nflds-3,localZ,localY,localX),dtype=np.double)
    for lz  in range(lblz):
        for ly in range(lbly):
            myoff = myoff1 + ly*nppb*fbp + lz*bly*nppb*fbp
            fh.Set_view(myoff,MPI.BYTE)
            fh.Read_all(uvwl)

            ins = 0
            ine = ins + lx1

            jns = 0 + ly*ly1
            jne = jns + ly1

            kns = 0 + lz*lz1
            kne = kns + lz1

            vx[kns:kne,jns:jne,ins:ine]=uvwl[0::nflds].reshape((lz1,ly1,lx1))
            vy[kns:kne,jns:jne,ins:ine]=uvwl[1::nflds].reshape((lz1,ly1,lx1))
            vz[kns:kne,jns:jne,ins:ine]=uvwl[2::nflds].reshape((lz1,ly1,lx1))
            for n in range(3,nflds):
                t[k,j,i,n-3] = uvwl[n::nflds].reshape(lz1,ly1,lx1)
            uvwl[:] = 0
    fh.Close()
    #if mesh is stored make sure to return it as an output
    if ifmesh == 'T':
        return vx,vy,vz, x, y, z, t
    else:
        return vx,vy,vz,t

