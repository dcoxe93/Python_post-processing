import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import pickle

from scipy.special import roots_legendre


Reynolds = 360
expir = 'OSC'

if Reynolds == 170:
    Re_bulk = 5000
elif Reynolds == 360:
    Re_bulk = 11700
else:
    Re_bulk = 25800

utau = 2*Reynolds/Re_bulk

if expir == 'STD':
    nphases = 1
    nfpf = 3200
elif expir == 'OSC':
    nphases = 32
    nfpf = 100 
ntot = nfpf*nphases
ptemp = 'RE{0}/RE{0}_{1}_{2}_{3}.pkl'

for ph in range(nphases):
    for fi in range(1,nfpf+1):
        pname = ptemp.format(Reynolds,expir,ph,fi)
        with open(pname,'rb') as f:
            data = pickle.load(f)
            data = data[0]
        if ph == 0 and fi == 1:
            nf,ny = data.shape
            tdata = np.zeros((ntot,nf,ny))
        ind = (fi-1)*nphases + ph

        tdata[ind] = data

r,w = roots_legendre(ny)

r = (1-(r+1)/2)*Reynolds
t = np.linspace(1,ntot,ntot)*1

R,T = np.meshgrid(r,t,indexing='xy')

f,a = plt.subplots(constrained_layout=True,figsize=(8,4))
pdata = tdata[:,7,:]/(utau**2)
z_max = 1.1*np.amax(pdata)
z_min = 1.1*np.amin(pdata)

cf = a.contourf(T,R,pdata,cmap=cm.coolwarm,linewidth=0)
cb = f.colorbar(cf,shrink=0.5,aspect=5)
#a.set_yscale('log')
a.set_ylim(1,Reynolds)

#a.set_xlim(t[0],t[-1]/4)


plt.show()
