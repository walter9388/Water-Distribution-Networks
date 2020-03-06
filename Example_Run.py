import pyWDN

filename = '25nodesData'
filename = 'BURWELMA_2019-12-25_-_2019-12-27'

temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)

# temp.make_network_graph()
# fig, ax = temp.G.plot_network()
# fig, ax = temp.G.plot_reservoirs(fig=fig, ax=ax)
# fig.legend()
# fig.show()
# temp.G.figshow_leaflet(fig)
