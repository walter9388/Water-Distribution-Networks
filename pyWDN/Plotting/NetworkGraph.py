import numpy as np
import networkx as nx
import scipy as sc
import matplotlib.pyplot as plt
import mplleaflet
import osmnx as ox
import folium
import seaborn as sns

class makenetworkgraph:

    def __init__(self, x, y, A,**kwargs):
        self.x = x
        self.y = y
        self.A = A
        G = ox.nx.MultiDiGraph()
        for i in range(len(x)):
            G.add_node(i, x=x[i], y=y[i], pos=(x[i],y[i]))
        for i in range(A.shape[0]):
            G.add_edge(int(sc.sparse.find(A[i, :] == 1)[1][0]), int(sc.sparse.find(A[i, :] == -1)[1][0]), name='asdfasd')
        self.G = G
        H0_nodes = kwargs.get('H0_nodes', None)
        if H0_nodes is not None:
            self.H0_nodes=H0_nodes


    def plot(self, outputtype, **kwargs):
        pos = ox.nx.get_node_attributes(self.G, 'pos')
        fig = plt.figure()
        plt.title('Pressure Simulation - date/time')
        node_size = kwargs.get('node_size', 0)
        node_color = kwargs.get('node_color', 'b')
        if type(node_color) != str:
            vmin = np.floor((min(node_color) / 10)) * 10
            vmax = np.ceil((max(node_color) / 10)) * 10
            ox.nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color, cmap=plt.cm.jet, vmin=vmin,
                    vmax=vmax)
            if outputtype == "leaflet":
                mplleaflet.show(fig)
            else:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
                sm._A = []
                cbar = plt.colorbar(sm)  # ,shrink=0.95)
                cbar.set_label('Pressure (mH20)', rotation=270, va='bottom')
                plt.show()

        else:
            ox.nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color)
            ox.nx.draw_networkx_nodes(self.G, pos, nodelist=[3,5],nodesize=10,node_color='r')
            fig.show()
            if outputtype == "leaflet":
                mplleaflet.show(fig)

    def plot_network(self,**kwargs):
        node_size = kwargs.get('node_size', 0)
        node_color = kwargs.get('node_color', 'b')
        fig = kwargs.get('fig', None)
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', 'network')
        if fig is None and ax is None:
            fig, ax = plt.subplots(1,1)
        elif fig is None:
            fig = plt.figure()

        pos = ox.nx.get_node_attributes(self.G, 'pos')
        ox.nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color,ax=ax,label=label)
        return fig, ax

    def plot_reservoirs(self,**kwargs):
        fig = kwargs.get('fig', None)
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', 'H0')
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1)
        elif fig is None:
            fig = plt.figure()
        pos = ox.nx.get_node_attributes(self.G, 'pos')
        ox.nx.draw_networkx_nodes(self.G, pos, nodelist=self.H0_nodes, nodesize=10, node_color='r',ax=ax,label=label)
        return fig, ax

    def plot_connected_points(self, closed_links=None, **kwargs):
        fig = kwargs.get('fig', None)
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', 'subgraph_')
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1)
        elif fig is None:
            fig = plt.figure()
        pos = ox.nx.get_node_attributes(self.G, 'pos')
        H = self.remove_links_from_G(closed_links)
        subgraphsets = list(ox.nx.weakly_connected_components(H))
        # subgraphsets = list(ox.nx.connected_components(ox.nx.Graph(H)))
        colours = self.get_colour_palette(len(subgraphsets))
        for ii in range(len(subgraphsets)):
            ox.nx.draw_networkx_nodes(H, pos, nodelist=list(subgraphsets[ii]), nodesize=4, ax=ax,
                                       node_color=np.array([colours[ii]]), label=label+str(ii))
        return fig, ax

    def remove_links_from_G(self, closed_links):
        if closed_links is not None:
            H = self.G.__class__()
            H.add_nodes_from(self.G)
            H.add_edges_from(self.G.edges)
            closed_links.sort(reverse=True)
            for ii in closed_links:
                H.remove_edge(*[np.where(self.A[ii, :].toarray() == 1)[1][0],
                                np.where(self.A[ii, :].toarray() == -1)[1][0], 0])
        else:
            H = self.G
        return H

    def get_colour_palette(self, n):
        return sns.color_palette("hls", n)

    def figshow_leaflet(self, fig):
        mplleaflet.show(fig)

    def make_folium(self,**kwargs):
        # filename = kwargs.get('filename','folium_output_.html')

        mapboxtoken = 'pk.eyJ1Ijoid2FsdGVyOTM4OCIsImEiOiJjazhzbmhocGUwMWEyM25uZDd1Z3hwYjA4In0.5JWjuw2ZyuHVzxnvNx1cfQ'
        # username_id='walter9388.ck8snlelx2cep1intbso64rza'
        username_id = 'mapbox.dark'

        self.G.graph = {'crs': 'WGS84',
                        'name': 'unnamed',
                        }

        map = folium.Map(location=[0, 0],
                         zoom_start=12,
                         tiles=None,
                         )
        folium.TileLayer(
            'http://api.mapbox.com/v4/' + username_id + '/{z}/{x}/{y}.png?access_token=' + mapboxtoken,
            attr='Mapbox | Source: InfraSense Labs | Written by Alex Waldron, 2020',
            name='Base Map',
        ).add_to(map)

        network_fg = folium.FeatureGroup(name='Distribution Mains')
        map.add_child(network_fg)
        graph_map = ox.plot_graph_folium(self.G,
                                         edge_width=1,
                                         edge_color='#FFFFFF',
                                         popup_attribute='name',
                                         # graph_map=map,
                                         )

        temp = list(graph_map._children.keys())[1:-1]
        [graph_map._children[temp[i]].add_to(network_fg) for i in range(len(temp))]
        map.fit_bounds(map.get_bounds())
        folium.LayerControl().add_to(map)

        self.folium_map = map
        # filepath = 'folium_output_.html'
        # map.save(filename)
