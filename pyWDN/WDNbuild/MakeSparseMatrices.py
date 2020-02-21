import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
from sksparse.cholmod import cholesky #if using windows, can install via https://github.com/xmlyqing00/Cholmod-Scikit-Sparse-Windows


def make_incidence_matrices(self):
    """based on the work of Edo Abraham (2014), Imperial College London"""

    iA12 = []
    jA12 = []
    sA12 = []

    iA10 = []
    jA10 = []
    sA10 = []

    for ip in range(self.np):
        try:
            startNode = self.NodeIdMap[self.pipes[ip]['startNodeId']]
        except:
            raise ValueError('startNodeId ' + self.pipes[ip]['startNodeId'] + ' is not in NodeIdMap')

        try:
            endNode = self.NodeIdMap[self.pipes[ip]['endNodeId']]
        except:
            raise ValueError('endNodeId ' + self.pipes[ip]['endNodeId'] + ' is not in NodeIdMap')

        if startNode < self.nn:
            iA12.append(ip)
            jA12.append(startNode)
            sA12.append(-1)
        else:
            iA10.append(ip)
            jA10.append(startNode - self.nn)
            sA10.append(-1)

        if endNode < self.nn:
            iA12.append(ip)
            jA12.append(endNode)
            sA12.append(1)
        else:
            iA10.append(ip)
            jA10.append(endNode - self.nn)
            sA10.append(1)

    A12 = sp.csc_matrix((sA12, (iA12, jA12)), shape=(self.np, self.nn))
    A10 = sp.csc_matrix((sA10, (iA10, jA10)), shape=(self.np, self.n0))
    return A12, A10

def make_null_space(A12,nn,npp,closed_pipes):
    if closed_pipes != []:
        A12[closed_pipes,:] = [];
        npp = npp - len(closed_pipes)

    nc=npp-nn
    (P, L, U) = la.lu(A12.toarray())
    P = sp.csc_matrix(P)
    L = sp.csc_matrix(L)
    U = sp.csc_matrix(U)

    L1 = L[:nn, :]
    L2 = L[nn:, :]
    K = -spla.spsolve(L1.T,L2.T)
    Z = P.T@sp.vstack([K,sp.eye(nc)])

    factor=cholesky(A12.T@A12)
    P_L=sp.eye(nn)
    Pr=sp.csc_matrix(P_L.toarray()[factor.P(), :])
    L_A12 = factor.L()

    nulldata={
        'Pr'    :   Pr,
        'L_A12' :   L_A12,
        'A12'   :   A12,
        'Z'     :   Z
    }
    auxdata={
        'closed_links'  :   closed_pipes
    }
    return nulldata, auxdata
