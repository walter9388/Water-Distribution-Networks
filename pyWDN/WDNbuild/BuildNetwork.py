from pyWDN.WDNbuild.MakeSparseMatrices import *
from pyWDN.Solvers.hydraulics import evaluate_hydraulic
from pyWDN.Plotting.NetworkGraph import makenetworkgraph
from pyWDN.Plotting.OS2WGS import OSGB36toWGS84
import scipy.sparse as sp
import os, re


class BuildWDN:
    def __init__(self, pipes, junctions, reservoirs, headloss_formula):
        # pipes
        self.pipes = pipes
        self.np = len(self.pipes)
        self.LinkIdMap = {self.pipes[i]['Id']: i for i in range(self.np)}
        self.L = np.array([self.pipes[i]['Length'] for i in range(self.np)])
        self.D = np.array([self.pipes[i]['Diameter'] for i in range(self.np)])
        self.C = np.array([self.pipes[i]['Roughness'] for i in range(self.np)])

        # junctions
        self.junctions = junctions
        self.nn = len(self.junctions)
        self.NodeIdMap = {self.junctions[i]['Id']: i for i in range(self.nn)}
        self.elev = np.array([self.junctions[i]['Elev'] for i in range(self.nn)])

        # reservoirs
        self.reservoirs = reservoirs
        self.n0 = len(self.reservoirs)
        self.NodeIdMap.update({self.reservoirs[i]['Id']: i + self.nn for i in range(self.n0)})

        # headloss formula
        self.headloss = {'formula': headloss_formula}
        if headloss_formula == 'H-W':
            self.headloss['n_exp'] = 1.852
        elif headloss_formula == 'D-W':
            self.headloss['n_exp'] = 2

        # A12 & A10
        self.A12, self.A10 = make_incidence_matrices(self)

        # Null Space
        self.closed_pipes = []
        try:
            self.remake_null_space()
        except:
            pass # look into this further

    def evaluate_hydraulics(self, print_timestep=True, **kwargs):
        # store temporarily the class attributes which are being replaced
        temp_dict = {key: self.__getattribute__(key) for key, value in kwargs.items()}
        # replace attributes with new attributes from kwargs
        self.__dict__.update((key, value) for key, value in kwargs.items())
        # evaluate hydraulics
        self.h_sim, self.q_sim = evaluate_hydraulic(self.A12, self.A10, self.C, self.D, self.demands, self.H0,
                                                    self.IndexValves, self.L, self.nn, self.np, self.nl, self.headloss,
                                                    self.nulldata, self.auxdata, print_timestep=print_timestep)
        # reset class attributes to origin values
        self.__dict__.update((key, value) for key, value in temp_dict.items())

    def make_network_graph(self):
        x = [self.junctions[i]['Coords'][0] for i in range(self.nn)] + [self.reservoirs[i]['Coords'][0] for i in range(self.n0)]
        y = [self.junctions[i]['Coords'][1] for i in range(self.nn)] + [self.reservoirs[i]['Coords'][1] for i in range(self.n0)]
        A = sp.hstack((self.A12, self.A10))
        if np.all(np.array(x+y)>1e4) and np.all(np.array(x+y)<1e6): # if OS coordinates:
            for i in range(len(x)):
                y[i], x[i] = OSGB36toWGS84(x[i], y[i])
        H0_nodes=[self.NodeIdMap[self.reservoirs[i]['Id']] for i in range(self.n0)]
        self.G = makenetworkgraph(x,y,A,H0_nodes=H0_nodes)

    def remake_null_space(self):
        self.nulldata, self.auxdata = make_null_space(self.A12, self.nn, self.np, self.closed_pipes)
        self.auxdata['max_iter'] = 50
        self.auxdata['kappa'] = 1e7
        self.auxdata['tol_err'] = 1e-6


class BuildWDN_fromMATLABfile(BuildWDN):
    def __init__(self, *filenamein):
        # define filename
        if len(filenamein) > 1:
            raise ValueError('only put the filename in')
        networks_root=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'NetworkFiles')
        networks_files=os.listdir(networks_root)
        pattern=('^(' + filenamein[0] + '|' + filenamein[0] + '.mat){1}$').encode('unicode_escape').decode()
        filebool=[re.match(pattern, networks_files[i]) is not None for i in range(len(networks_files))]
        if np.any(filebool):
            filename=os.path.join(networks_root,networks_files[filebool.index(True)])
        else:
            filename=filenamein[0]

        # import .mat file
        import scipy.io as sc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmf = sc.loadmat(filename)
        self._tmf = tmf

        # define pipes
        p = tmf['pipes'][0]
        # pipes = [
        #     {
        #         list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
        #         p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
        #     }
        #     for j in range(len(p))
        # ]
        pipes=[]
        for j in range(len(p)):
            tempdic={}
            for i in range(len(list(p[j][0].dtype.fields.keys()))):
                tempdickey=list(p[j][0].dtype.fields.keys())[i]
                if len(p[j][0][0][i])==0:
                    tempdic[tempdickey]=''
                elif isinstance(p[j][0][0][i][0], str):
                    tempdic[tempdickey] = p[j][0][0][i][0]
                elif tempdickey == 'Diameter':
                    tempdic[tempdickey] = p[j][0][0][i][0][0] / 1000 # mm -> m
                else:
                    tempdic[tempdickey] = p[j][0][0][i][0][0]
            # check if length, roughness and diameter are included
            if 'Roughness' not in tempdic:
                tempdic['Roughness'] = tmf['C'][j, 0]
            if 'Length' not in tempdic:
                tempdic['Length'] = tmf['L'][j, 0]
            if 'Diameter' not in tempdic:
                tempdic['Diameter'] = tmf['D'][j, 0]
            pipes.append(tempdic)

        # define junctions
        p = tmf['junctions'][0]
        # couldn't handle empty keys (e.g. Patterns):
        # junctions = [
        #     {
        #         list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
        #         p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
        #     }
        #     for j in range(len(p))
        # ]
        junctions=[]
        for j in range(len(p)):
            tempdic={}
            for i in range(len(list(p[j][0].dtype.fields.keys()))):
                tempdickey=list(p[j][0].dtype.fields.keys())[i]
                if len(p[j][0][0][i])==0:
                    tempdic[tempdickey]=''
                elif isinstance(p[j][0][0][i][0], str):
                    tempdic[tempdickey] = p[j][0][0][i][0]
                else:
                    tempdic[tempdickey] = p[j][0][0][i][0][0]
            junctions.append(tempdic)


        # define reservoirs
        p = tmf['reservoirs'][0]
        reservoirs = []
        for j in range(len(p)):
            tempdic = {}
            for i in range(len(list(p[j][0].dtype.fields.keys()))):
                tempdickey = list(p[j][0].dtype.fields.keys())[i]
                if len(p[j][0][0][i]) == 0:
                    tempdic[tempdickey] = ''
                elif isinstance(p[j][0][0][i][0], str):
                    tempdic[tempdickey] = p[j][0][0][i][0]
                else:
                    tempdic[tempdickey] = p[j][0][0][i][0][0]
            reservoirs.append(tempdic)

        # headloss fucntions
        n_exp = tmf['n_exp']
        if np.all(n_exp == 1.852):
            headloss_formula = 'H-W'
        elif np.all(n_exp == 2):
            headloss_formula = 'D-W'
        else:
            warnings.warn('don\'t know what headloss formula to use, so using H-W')
            headloss_formula = 'H-W'

        # Build WDN
        super().__init__(pipes, junctions, reservoirs, headloss_formula)

        # add xy coordinates to junctions and reservoirs dicts
        for j in range(self.nn):
            self.junctions[j].update(
                {
                    'Coords': tuple(map(float,tmf['junctionXYData'][0][i][0].split()[1:]))
                    for i in range(len(tmf['junctionXYData'][0])) if
                    tmf['junctionXYData'][0][i][0].split()[0] == self.junctions[j]['Id']
                }
            )
        for j in range(self.n0):
            self.reservoirs[j].update(
                {
                    'Coords': tuple(map(float,tmf['junctionXYData'][0][i][0].split()[1:]))
                    for i in range(len(tmf['junctionXYData'][0])) if
                    tmf['junctionXYData'][0][i][0].split()[0] == self.reservoirs[j]['Id']
                }
            )

        # define times
        p = tmf['times'][0]
        self.times = [
            {
                list(p.dtype.fields.keys())[i]: p[0][i][0] if isinstance(p[0][i][0], str) else
                p[0][i][0][0] for i in range(len(list(p.dtype.fields.keys())))
            }
            for j in range(len(p))
        ]

        # define valves
        if tmf['valves'].size == 0:
            self.valves = []
        else:
            p = tmf['valves'][0]
            self.valves = [
                {
                    list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
                    p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
                }
                for j in range(len(p))
            ]

        # nl
        self.nl = int(tmf['nl'][0][0])

        #PRVs/BVs/indexvalves
        self.PRVs = (list(tmf['PRVs'][0]-1) if len(tmf['PRVs']) != 0 else [])
        self.BVs = (list(tmf['BVs'][0]-1) if len(tmf['BVs']) != 0 else [])
        self.IndexValves = self.PRVs + self.BVs

        # add all remaining variables in tmf as attributes to the WDN class
        dontbuild=['n_exp','junctionXYData']
        for i in range(tmf.keys().__len__()):
            if list(tmf.keys())[i] not in self.__dir__() and list(tmf.keys())[i] not in dontbuild:
                attr_name = list(tmf.keys())[i]
                self.__setattr__(attr_name, tmf[attr_name])

