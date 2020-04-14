import pyWDN

filename = '25nodesData'
filename = 'BURWELMA_2019-12-25_-_2019-12-27'
# filename = r'C:\Users\walte\Documents\python\Water-Distribution-Networks\pyWDN\NetworkFiles\Daily\2018_06_06\BWFLnet_DMA1_06_06_2018__07_06_2018.mat'

temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)



###### graphing tests:
temp.make_network_graph()

# folium
temp.G.make_folium()
temp.G.folium_map.save('folium_'+filename+'.html')

# matplotib
fig, ax = temp.G.plot_network()
fig, ax = temp.G.plot_reservoirs(fig=fig, ax=ax)
fig.show()

