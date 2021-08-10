import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.special import roots_legendre
from matplotlib import rc
import matplotlib
#matplotlib.use("Agg")

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text',usetex=True)

expirs = ['STD','OSC']
Reynolds = 170
if Reynolds == 170:
    Re_bulk = 5000
elif Reynolds == 360:
    Re_bulk = 11700
else:
    Re_bulk = 25800

utau = 2*Reynolds/Re_bulk
pickleN = 'RE{0}/RE{0}_{1}_{2}.pkl'
for e in range(2):
    if e == 0:
        nph = 1
    else:
        nph = 32

    for ph in range(nph):
        with open(pickleN.format(Reynolds,expirs[e],ph),'rb') as f:
            data = pickle.load(f)
            data = data[0]
            data[:-1:] = data[:-1:]/utau
            data[3:-1:]=data[3:-1:]/utau

        if e == 0 and ph == 0:
            nflds,ny = data.shape
            statsLTA = np.zeros((2,nflds,ny),dtype=np.double)
        beta = 1/(ph+1)
        alpha = 1-beta
        statsLTA[e] = statsLTA[e]*alpha + beta*data

dpdx = 4*utau**2
nu = 1/Re_bulk

labels = (['u_x','u_r','u_\\theta',
           '{u_x^\\prime}^+ {u_x^\\prime}^+','{u_r^\\prime}^+ {u_r^\\prime}^+','{u_\\theta^\\prime}^+ {u_\\theta^\\prime}^+',
           '{u_x^\\prime}^+ {u_r^\\prime}^+','{u_r^\\prime}^+ {u_\\theta^\\prime}^+','{u_\\theta^\\prime}^+ {u_x^\\prime}^+'])
f_ind = (['u','v','w','uu','vv','ww','uv','vw','wu'])
x,w = roots_legendre(ny)
r = (x+1)/4
y = (.5 - r)*2*Reynolds
en = [0.,0.]
for flds in [3,4,5]:
    fig,ax = plt.subplots(constrained_layout=True,figsize=(14,10))
    s_max = np.zeros((2,),dtype=np.double)
    for e in range(2):
        p = statsLTA[e][flds]
        
        en[e]+=((.5/2)*np.dot(p,r*w))
        if flds > 2 and flds < 6:
            p = np.sqrt(p)
        if e % 2 == 0:
            ax.plot(y,p,color='k',linewidth=2,linestyle='solid')
        else:
            ax.plot(y,p,color='b',linewidth=2,linestyle='dashed')

        s_max[e] = np.max(1.1*p)

    #ax.set_xscale('log')
    ax.set_xlim(1,Reynolds)
    ax.set_xlabel('$y^+$',fontsize=30)

    ax.set_ylim(0,np.max(s_max))
    ax.set_ylabel('$\\langle {0} \\rangle_{{\\theta,x,t}}$'.format(labels[flds]),fontsize=30)
    if flds > 2 and flds < 6:
        ax.set_ylabel('$\\sqrt{{ \\langle {0} \\rangle_{{\\theta,x,t}} }}$'.format(labels[flds]),fontsize=30)
    

    ax.tick_params(axis='both',which='major',labelsize=16,width=2)
    ax.tick_params(axis='both',which='minor',width=1)

    plt.show()
    plt.savefig('{0}stress_{1}.png'.format(f_ind[flds],Reynolds),dpi=300,format='png')
    plt.savefig('{0}stress_{1}.pdf'.format(f_ind[flds],Reynolds),dpi=300,format='pdf')
    plt.close()

print(en)
k = np.zeros((2,ny))
s_max = np.zeros((2,))
for e in range(2):
    for flds in [3,4,5]:
        k[e] += statsLTA[e][flds]/4
    s_max[e] = np.max(1.1*k[e])
print(s_max)
fig,ax = plt.subplots(constrained_layout=True)

ax.plot(y,k[0],color='k',linewidth=2,linestyle='solid')
ax.plot(y,k[1],color='b',linewidth=2,linestyle='dashed')

ax.set_xscale('log')
ax.set_xlim(1,Reynolds)
ax.set_xlabel('$y^+$',fontsize=30)

ax.set_ylim(0,s_max[1])
ax.set_ylabel('$ k^2 = \\frac{1}{4}\\langle' \
                '(u_i^\\prime)^2'\
                '\\rangle_{\\theta,x,t}$',fontsize=30)

ax.tick_params(axis='both',which='major',labelsize=40,width=6,length=12)
ax.tick_params(axis='both',which='minor',width=3,length=6)

