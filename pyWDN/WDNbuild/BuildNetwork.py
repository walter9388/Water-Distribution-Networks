from pyWDN.WDNbuild.MakeSparseMatrices import *
from pyWDN.Solvers.hydraulics import evaluate_hydraulic


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
        self.nulldata, self.auxdata = make_null_space(self.A12, self.nn, self.np, self.closed_pipes)
        self.auxdata['max_iter'] = 50
        self.auxdata['kappa'] = 1e7
        self.auxdata['tol_err'] = 1e-6

    def evaluate_hydraulics(self):

        H_sim, Q_sim = evaluate_hydraulic(self)


class BuildWDN_fromMATLABfile(BuildWDN):
    def __init__(self, filename):
        # import .mat file
        import scipy.io as sc
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tmf = sc.loadmat(filename)
        self.tmf = tmf

        # define pipes
        p = tmf['pipes'][0]
        pipes = [
            {
                list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
                p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
            }
            for j in range(len(p))
        ]

        # define junctions
        p = tmf['junctions'][0]
        junctions = [
            {
                list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
                p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
            }
            for j in range(len(p))
        ]

        # define reservoirs
        p = tmf['reservoirs'][0]
        reservoirs = [
            {
                list(p[j][0].dtype.fields.keys())[i]: p[j][0][0][i][0] if isinstance(p[j][0][0][i][0], str) else
                p[j][0][0][i][0][0] for i in range(len(list(p[j][0].dtype.fields.keys())))
            }
            for j in range(len(p))
        ]

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
                    'Coords': tuple(tmf['junctionXYData'][0][i][0].split()[1:])
                    for i in range(len(tmf['junctionXYData'][0])) if
                    tmf['junctionXYData'][0][i][0].split()[0] == self.junctions[j]['Id']
                }
            )
        for j in range(self.n0):
            self.reservoirs[j].update(
                {
                    'Coords': tuple(tmf['junctionXYData'][0][i][0].split()[1:])
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
        self.nl = tmf['nl'][0][0]

        #PRVs/BVs/indexvalves
        self.PRVs = list(tmf['PRVs'])
        self.BVs = list(tmf['BVs'])
        self.IndexValves = self.PRVs + self.BVs

        # add all remaining variables in tmf as attributes to the WDN class
        for i in range(tmf.keys().__len__()):
            if list(tmf.keys())[i] not in self.__dir__() and list(tmf.keys())[i] != 'n_exp':
                attr_name = list(tmf.keys())[i]
                self.__setattr__(attr_name, tmf[attr_name])
