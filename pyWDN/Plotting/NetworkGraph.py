import numpy as np
import networkx as nx
import scipy as sc
import matplotlib.pyplot as plt
import mplleaflet

class makenetworkgraph:

    def __init__(self, x, y, A,**kwargs):
        self.x = x
        self.y = y
        self.A = A
        G = nx.DiGraph()
        for i in range(len(x)):
            G.add_node(i, pos=(x[i], y[i]))
        for i in range(A.shape[0]):
            G.add_edge(int(sc.sparse.find(A[i, :] == 1)[1][0]), int(sc.sparse.find(A[i, :] == -1)[1][0]))
        self.G = G
        H0_nodes = kwargs.get('H0_nodes', None)
        if H0_nodes is not None:
            self.H0_nodes=H0_nodes


    def plot(self, outputtype, **kwargs):
        pos = nx.get_node_attributes(self.G, 'pos')
        fig = plt.figure()
        plt.title('Pressure Simulation - date/time')
        node_size = kwargs.get('node_size', 0)
        node_color = kwargs.get('node_color', 'b')
        if type(node_color) != str:
            vmin = np.floor((min(node_color) / 10)) * 10
            vmax = np.ceil((max(node_color) / 10)) * 10
            nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color, cmap=plt.cm.jet, vmin=vmin,
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
            nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color)
            nx.draw_networkx_nodes(self.G, pos, nodelist=[3,5],nodesize=10,node_color='r')
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

        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw(self.G, pos, node_size=node_size, arrows=False, node_color=node_color,ax=ax,label=label)
        return fig, ax

    def plot_reservoirs(self,**kwargs):
        fig = kwargs.get('fig', None)
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', 'H0')
        if fig is None and ax is None:
            fig, ax = plt.subplots(1, 1)
        elif fig is None:
            fig = plt.figure()
        pos = nx.get_node_attributes(self.G, 'pos')
        nx.draw_networkx_nodes(self.G, pos, nodelist=self.H0_nodes, nodesize=10, node_color='r',ax=ax,label=label)
        return fig, ax


    def figshow_leaflet(self,fig):
        mplleaflet.show(fig)

