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
    if len(closed_pipes) > 0:
        CLmask = np.ones(npp, dtype=bool)
        CLmask[closed_pipes] = False
        A12 = A12[CLmask, :]
        npp = npp - len(closed_pipes)
    nc = npp-nn

    # # lu method
    # (P, L, U) = la.lu(A12.toarray())
    # # P = sp.csc_matrix(P)
    # L = sp.csr_matrix(L)
    # # U = sp.csc_matrix(U)
    # L1 = L[:nn, :]
    # L2 = L[nn:, :]
    # K = -spla.spsolve(L1.T,L2.T)
    # Z = P.T@sp.vstack([K,sp.eye(nc)])

    # Elhay null bases
    Pt, _, T = permute_cotree(A12)
    L21 = -spla.spsolve(T[:nn, :nn].T, T[nn:npp, :nn].T).T
    Z = Pt.T@sp.hstack([L21, sp.eye(nc)]).T

    # cholseky
    factor=cholesky((A12.T@A12).tocsc())
    P_L = sp.eye(nn)
    Pr = sp.csc_matrix(P_L.toarray()[factor.P(), :])
    L_A12 = factor.L()

    nulldata={
        'Pr'    :   Pr,
        'L_A12' :   L_A12,
        'A12'   :   A12,
        'Z'     :   Z
    }
    auxdata={
        'closed_pipes'  :   closed_pipes,
        'max_iter'      :   50,
        'kappa'         :   1e7,
        'tol_err'       :   1e-6
    }
    return nulldata, auxdata

def permute_cotree(A):
    """based on the work of Edo Abraham (2014), Imperial College London"""

    n, m = A.shape
    Pt = sp.eye(n)
    Rt = sp.eye(m)

    for i in range(m):
        K = A[i:, i:]
        try:
            r=np.where(np.sum(abs(K), 1) == 1)[0][0]
        except:
            r = np.array([], dtype=int)
        c = np.where((abs(K[r, :]) == 1).toarray())[1]

        if r!=0: # don't do operations unless row permutations are needed
            P = sp.eye(n).tolil()
            P[i, i] = 0
            P[i + r, i + r] = 0
            P[i, i + r] = 1
            P[i + r, i] = 1
            Pt = P @ Pt
            A = P @ A

        if c!=0: # don't do operations unless column permutations are needed
            R = sp.eye(m).tolil()
            R[i, i] = 0
            R[i + c, i + c] = 0
            R[i, i + c] = 1
            R[i + c, i] = 1
            Rt = R @ Rt
            A = A @ R

    return Pt.tocsr(), Rt.tocsr(), A
