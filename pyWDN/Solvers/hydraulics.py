import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import psutil

def evaluate_hydraulic(A12,A10,C,D,demands,H0,IndexValves,L,nn,NP,nl,headloss,nulldata,auxdata):
    H = np.empty((nn,nl)) * np.nan
    Q = np.empty((NP,nl)) * np.nan
    err = np.zeros(nl)
    iter = np.zeros(nl)
    CONDS = np.zeros(nl)
    check = np.zeros(nl)
    for kk in range(nl):
        xtemp = spla.spsolve(nulldata['L_A12'],nulldata['Pr'] @ demands[:,kk])
        w = nulldata['Pr'].T @ spla.spsolve(nulldata['L_A12'].T,xtemp)
        xtemp = A12 @ w
        nulldata['x'] = xtemp
        q_0 = 0.3 * np.ones((NP, 1))
        h_0 = 130 * np.ones((nn, 1))
        if headloss['formula']=='H-W':
            auxdata['ResCoeff'] = 10.670 * L / (C ** headloss['n_exp'] * (D/1000) ** 4.871)
            auxdata['ResCoeff'][IndexValves] = (8 / (np.pi ** 2 * 9.81)) * (D[IndexValves]/1000) ** -4 * C[IndexValves]
            if 'Eta' in nulldata:
                Eta = nulldata['Eta']
                A13 = nulldata['A13']
                Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace(A12, A12.T, A10, A13, H0[:,kk], Eta[:,kk], headloss['n_exp'], q_0, h_0, demands[:,kk], NP, nulldata, auxdata)
            else:
                Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace(A12, A12.T, A10, [], H0[:,kk], [], headloss['n_exp'], q_0, h_0, demands[:,kk], NP, nulldata, auxdata)
        elif headloss['formula']=='D-W':
            pass
            # auxdatDIAMETERS = datDIAMETERS;
            # auxdatROUGHNESS = zeros(np, 1);
            # auxdatROUGHNESS(setdiff(1: datNP, datIndexValves))=1e-3 * u(setdiff(1: datNP, datIndexValves));
            # auxdatROUGHNESS(datIndexValves) = u(datIndexValves);
            #.. auxdatIndexValves = datIndexValves;
            # auxdatLENGTHS = datLENGTHS;
            # n_exp = 2 * ones(np, 1);
            # if isfield(data, 'Eta')
            # Eta = datEta;
            # A13 = datA13;
            # [qtemp, htemp, err(kk), iter(kk), CONDS(kk), ERRORS(kk)] = minor_losses_solveDW_Nullspace_Valves_QuadraticConvergence(
            #     A12, A12
            # ',A10,A13,H0(:,kk),Eta(:,kk),n_exp,q_0,h_0,demands(:,kk),np,nulldata,auxdata);
            # else
            # [qtemp, htemp, err(kk), iter(kk), CONDS(kk), ERRORS(kk)] = minor_losses_solveDW_Nullspace_Valves_QuadraticConvergence(
            #     A12, A12
            # ',A10,[],H0(:,kk),[],n_exp,q_0,h_0,demands(:,kk),np,nulldata,auxdata);

        print('Time Step: %i'%(kk+1))
        # if ERRORS > auxdata['tol_err']:
        #     check[kk] = 1

    return H, Q


def solveHW_Nullspace(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, NP, nulldata, auxdata):
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
    import time
    filename = '25nodesData'
    temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    for _ in range(5):
        t = time.time()
        temp.evaluate_hydraulics()
        elapsed = time.time() - t
        print('Python: ' + str(elapsed)) # MATLAB runs the same network simulation in 0.155s

    from pyWDN.Solvers.call_ju import *

    # call_julia('hydraulics_ju.jl', 'funfun', temp.A12)
    for _ in range(5):
        t = time.time()
        H,Q = call_julia('hydraulics_ju.jl', 'evaluate_hydraulics', temp.A12, temp.A10, temp.C, temp.D, temp.demands, temp.H0, temp.IndexValves, temp.L, temp.nn, temp.np, temp.nl, temp.headloss, temp.nulldata, temp.auxdata)
        elapsed2 = time.time() - t
        print('Julia: ' + str(elapsed2))
