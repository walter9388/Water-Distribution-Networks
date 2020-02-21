import os
import matplotlib.pyplot as plt
import pyWDN

filename = os.path.join(os.getcwd(), 'pyWDN\\NetworkFiles\\25nodesData.mat')
temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)


fig, ax = plt.subplots(1,1)
ax.spy(temp.tmf['A12'])
fig.show()

fig1, ax1 = plt.subplots(1,1)
ax1.spy(temp.A12)
fig1.show()

fig2, ax2 = plt.subplots(1,1)
ax2.spy(temp.A10)
fig2.show()