import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp

def evaluate_hydraulic(a):#nn,np,nl,demands,headloss,nulldata,auxdata):
    H = np.empty((a.nn,a.nl)) * np.nan
    Q = np.empty((a.np,a.nl)) * np.nan
    err = np.zeros(a.nl)
    iter = np.zeros(a.nl)
    CONDS = np.zeros(a.nl)
    check = np.zeros(a.nl)
    for kk in range(a.nl):
        xtemp = spla.spsolve(a.nulldata['L_A12'],a.nulldata['Pr'] @ a.demands[:,kk])
        w = a.nulldata['Pr'].T @ spla.spsolve(a.nulldata['L_A12'].T,xtemp)
        xtemp = a.A12 @ w
        a.nulldata['x'] = xtemp
        q_0 = 0.3 * np.ones((a.np, 1))
        h_0 = 130 * np.ones((a.nn, 1))
        if a.headloss['formula']=='H-W':
            a.auxdata['ResCoeff'] = 10.670 * a.L / (a.C ** a.headloss['n_exp'] * (a.D/1000) ** 4.871)
            a.auxdata['ResCoeff'][a.IndexValves] = (8 / (np.pi ** 2 * 9.81)) * (a.D[a.IndexValves]/1000) ** -4 * a.C[a.IndexValves]
            if 'Eta' in a.nulldata:
                Eta = a.nulldata['Eta']
                A13 = a.nulldata['A13']
                Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace_Valves(a.A12, a.A12.T, a.A10, A13, a.H0[:,kk], Eta[:,kk], a.headloss['n_exp'], q_0, h_0, a.demands[:,kk], a.np, a.nulldata, a.auxdata)
            else:
                Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace_Valves(a.A12, a.A12.T, a.A10, [], a.H0[:,kk], [], a.headloss['n_exp'], q_0, h_0, a.demands[:,kk], a.np, a.nulldata, a.auxdata)
        elif a.headloss['formula']=='D-W':
            pass
            # auxdata.DIAMETERS = data.DIAMETERS;
            # auxdata.ROUGHNESS = zeros(np, 1);
            # auxdata.ROUGHNESS(setdiff(1: data.NP, data.IndexValves))=1e-3 * u(setdiff(1: data.NP, data.IndexValves));
            # auxdata.ROUGHNESS(data.IndexValves) = u(data.IndexValves);
            #.. auxdata.IndexValves = data.IndexValves;
            # auxdata.LENGTHS = data.LENGTHS;
            # n_exp = 2 * ones(np, 1);
            # if isfield(data, 'Eta')
            # Eta = data.Eta;
            # A13 = data.A13;
            # [qtemp, htemp, err(kk), iter(kk), CONDS(kk), ERRORS(kk)] = minor_losses_solveDW_Nullspace_Valves_QuadraticConvergence(
            #     A12, A12
            # ',A10,A13,H0(:,kk),Eta(:,kk),n_exp,q_0,h_0,demands(:,kk),np,nulldata,auxdata);
            # else
            # [qtemp, htemp, err(kk), iter(kk), CONDS(kk), ERRORS(kk)] = minor_losses_solveDW_Nullspace_Valves_QuadraticConvergence(
            #     A12, A12
            # ',A10,[],H0(:,kk),[],n_exp,q_0,h_0,demands(:,kk),np,nulldata,auxdata);

        print('Time Step: %i'%(kk+1))
        # if ERRORS > a.auxdata['tol_err']:
        #     check[kk] = 1

    return H, Q


def solveHW_Nullspace_Valves(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, NP, nulldata, auxdata):
    '''Based on the work of Edo Abraham, Imperial College London, 2014'''

    ResCoeff = auxdata['ResCoeff']
    CL = auxdata['closed_pipes']
    if len(CL) > 0:
        CLmask = np.ones(NP, dtype=bool)
        CLmask[CL] = False
        A12 = A12[CLmask, :]
        A21 = A12.T
        A10 = A10[CLmask, :]
        NP = NP - len(CL)
        A13 = A13[CLmask, :]
        ResCoeff = ResCoeff[CLmask, :]
        qk = qk[CLmask, :]

    x = nulldata['x']
    Z = nulldata['Z']
    P_r = nulldata['Pr']
    L_A12 = nulldata['L_A12']

    CONDS = 1
    ERRORS = 1

    max_iter = auxdata['max_iter']
    tol = auxdata['tol_err']
    kappa = auxdata['kappa']

    Method = 'standard'
    if len(A13)==0:
        eta = np.zeros((1,1))
        A13 = sp.csc_matrix((NP, 1))
    elif len(eta)==0 and 'TargetNodes' in auxdata:
        Method = 'fix_headTarget'
        TargetNodes = auxdata['TargetNodes']
        h_target = auxdata['h_target']
        valves = auxdata['valves']
        eta = np.zeros((len(TargetNodes), 1))


    G = ResCoeff[:,np.newaxis] * abs(qk) ** (n_exp - 1)
    err1 = np.linalg.norm(np.vstack((G * qk + A12 @ hk + A10 @ h0[:,np.newaxis] + A13 @ eta, A21 @ qk - d[:,np.newaxis])), ord=np.inf)

    nc = Z.shape[1]

    nnz_ZZ = (Z.T@sp.eye(NP)@Z).count_nonzero()
    ZT = Z.T

    Fdiag_old = sp.csc_matrix((NP, 1))
    X = sp.csc_matrix((nc, nc))

    updates=np.arange(NP)
    n_upd = len(updates)

    for kk in range(max_iter):
        Fdiag = n_exp * G
        sigma_max = np.max(Fdiag)
        t_k = np.maximum((sigma_max / kappa) - Fdiag, 0)
        Fdiag = Fdiag + t_k

        X = ZT @ sp.spdiags(Fdiag.T,[0],n_upd,n_upd) @ Z

        b = ZT@((Fdiag-G)*qk-A10@h0[:,np.newaxis]-A13@eta-Fdiag*x[:,np.newaxis])

        v = spla.spsolve(X,b)

        q = x[:,np.newaxis] + Z @ v[:,np.newaxis]

        b = A21@((Fdiag-G)*qk-A10@h0[:,np.newaxis]-A13@eta-Fdiag*q)
        y=spla.spsolve(L_A12,P_r @ b)
        h=P_r.T*spla.spsolve(L_A12.T,y)
        if Method == 'fix_headTarget':
            h_ideal=h
            h_ideal[TargetNodes]=h_target
            eta=-A12[valves,:]@h_ideal - A10[valves,:]@h0 - q[valves] * ResCoeff[valves,:] * abs(q[valves]) ** (n_exp - 1)

        G[updates] = ResCoeff[updates,np.newaxis] * abs(q[updates]) ** (n_exp - 1)
        err1 = np.linalg.norm(np.vstack((G * q + A12 @ h[:, np.newaxis] + A10 @ h0[:, np.newaxis] + A13 @ eta, A21 @ q - d[:, np.newaxis])), ord=np.inf)

        ERRORS = err1

        if err1 < tol:
            break
        else:
            qk = q

    if len(CL)>0:
        qorig = np.zeros((len(CL) + NP, 1))
        s = set(CL)
        qorig[[x for x in np.arange(len(CL)+NP) if x not in s]] = q
        q = qorig

    check = int(ERRORS > tol)

    return q.flatten(),h,err1,kk,CONDS,check


if __name__=='__main__':
    import os
    import pyWDN
    filename = os.path.join(os.path.dirname(os.getcwd()), 'NetworkFiles\\25nodesData.mat')
    temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    H,Q = evaluate_hydraulic(temp)
