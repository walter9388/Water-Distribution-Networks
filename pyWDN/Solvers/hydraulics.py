import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import psutil
from datetime import datetime as dt
from datetime import timedelta as td


def get_scaled_demand(demand,total_flow,flow_dt):
    if len(flow_dt) != len(total_flow):
        raise ValueError('flow and datetime should be the same length')
    int_secs=(60*60*24)/demand.shape[1]
    mult_dt=np.arange(dt(2020, 1, 1), dt(2020, 1, 2), td(seconds=int_secs), dtype=dt)
    mult_dt=[(mult_dt[i].hour, mult_dt[i].minute) for i in range(len(mult_dt))]
    flow_dt=[(flow_dt[i].hour, flow_dt[i].minute) for i in range(len(flow_dt))]
    demand_ts=[np.where((flow_dt[i]==np.array(mult_dt)).all(axis=1))[0][0] for i in range(len(flow_dt))]
    demand_sum=demand.sum(axis=0)

    new_demands=np.zeros((demand.shape[0],len(flow_dt)))
    for i in range(len(flow_dt)):
        new_demands[:,i] = demand[:,demand_ts[i]]*total_flow[i]/demand_sum[demand_ts[i]]

    return new_demands



def evaluate_hydraulic(A12, A10, C, D, demands, H0, IndexValves, L, nn, NP, nl, headloss, nulldata, auxdata, print_timestep=True):
    # for this to work all inputs must be in SI units (i.e. m, m^3/s, etc.)

    if H0.shape[0] == 0:
        raise AttributeError('There are no "H0" nodes in the network, therefore the hydraulics cannot be evaluated')

    H = np.empty((nn, nl)) * np.nan
    Q = np.empty((NP, nl)) * np.nan
    err = np.zeros(nl)
    iter = np.zeros(nl)
    CONDS = np.zeros(nl)
    check = np.zeros(nl)
    if "Eta" in nulldata:
        Eta = nulldata['Eta']
        A13 = nulldata['A13']
    else:
        Eta = np.zeros((0, nl))
        A13 = np.zeros((0, nl))

    for kk in range(nl):
        xtemp = spla.spsolve(nulldata['L_A12'], nulldata['Pr'] @ demands[:, kk])
        w = nulldata['Pr'].T @ spla.spsolve(nulldata['L_A12'].T, xtemp)
        xtemp = A12 @ w
        nulldata['x'] = xtemp
        q_0 = 0.3 * np.ones((NP, 1))
        h_0 = 130 * np.ones((nn, 1))
        if headloss['formula'] == 'H-W':
            auxdata['ResCoeff'] = 10.670 * L / (C ** headloss['n_exp'] * D ** 4.871)
            auxdata['ResCoeff'][IndexValves] = (8 / (np.pi ** 2 * 9.81)) * D[IndexValves] ** -4 * C[IndexValves]
            # # if 'Eta' in nulldata:
            # #     Eta = nulldata['Eta']
            # #     A13 = nulldata['A13']
            # Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace(A12, A12.T, A10, A13, H0[:,kk], Eta[:,kk], headloss['n_exp'], q_0, h_0, demands[:,kk], NP, nulldata, auxdata)
            # # else:
            # #     Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace(A12, A12.T, A10, [], H0[:,kk], [], headloss['n_exp'], q_0, h_0, demands[:,kk], NP, nulldata, auxdata)
        elif headloss['formula'] == 'D-W':
            auxdata['D'] = D[:, np.newaxis]
            auxdata['C'] = C[:, np.newaxis]
            auxdata['L'] = L[:, np.newaxis]
            auxdata['ResCoeff'] = np.zeros((NP, 1))
            # Q[:, kk], H[:, kk], err[kk], iter[kk], CONDS[kk], check[kk] = DW_temp(A12, A12.T, A10, A13,
            #                                                                           H0[:, kk], Eta[:, kk],
            #                                                                           headloss['n_exp'], q_0, h_0,
            #                                                                           demands[:, kk], NP,
            #                                                                           nulldata, auxdata)

        Q[:, kk], H[:, kk], err[kk], iter[kk], CONDS[kk], check[kk] = solve_Nullspace(A12, A12.T, A10, A13,
                                                                                      H0[:, kk], Eta[:, kk],
                                                                                      headloss['n_exp'], q_0, h_0,
                                                                                      demands[:, kk], NP,
                                                                                      nulldata, auxdata,
                                                                                      headloss['formula'])

        if print_timestep:
            print('Time Step: %i' % (kk + 1))
        # if ERRORS > auxdata['tol_err']:
        #     check[kk] = 1

    return H, Q


def solve_Nullspace(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, NP, nulldata, auxdata, headloss):
    '''Based on the work of Edo Abraham, Imperial College London, 2014'''
    if headloss == 'D-W':
        rho = 998.2  # density of water(kg / m ^ 3) at 20 degrees C
        mu = 1.002e-3  # dynamic viscosity(kg / m / s) at 20 degrees C
        Viscos = mu / rho  # kinematic viscosity(m ^ 2 / s) % viscosity of water at 20 deg C in m ^ 2 / s
        pD = auxdata['D']  # Pipe Diameters
        pR = auxdata['C']  # Pipe Roughness
        pL = auxdata['L']  # Pipe Length
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
        if headloss == 'D-W':
            pD = pD[CLmask, :]
            pR = pR[CLmask, :]
            pL = pL[CLmask, :]

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
    if len(A13) == 0:
        eta = np.zeros((1, 1))
        A13 = sp.csc_matrix((NP, 1))
    elif len(eta) == 0 and 'TargetNodes' in auxdata:
        Method = 'fix_headTarget'
        TargetNodes = auxdata['TargetNodes']
        h_target = auxdata['h_target']
        valves = auxdata['valves']
        eta = np.zeros((len(TargetNodes), 1))

    # calculate G (and constants for D-W)
    if headloss == 'D-W':
        cc = 5.74 * (np.pi * Viscos / 4) ** 0.9
        ResCoeff_LAMIflow = (128 * Viscos / (np.pi * 9.81)) * pL * (pD ** -4)
        K = pL / (pD ** 5)

        ResCoeff = np.zeros((NP, 1))
        G = np.zeros((NP, 1))
        Fdiag = np.ones((NP, 1))

        # cubic interpolating spline
        theta_hat = ((1 / 3.7) * pR) / pD + (5.74 / 4000 ** 0.9)
        tau = 0.00514215
        xi = -0.86859
        alphaterm = xi ** 2 * np.log(theta_hat) ** 2
        betaterm = xi ** 3 * np.log(theta_hat) ** 3
        alpha = np.hstack([
            5 / alphaterm,
            0.128 - 12 / alphaterm,
            -0.128 + 9 / alphaterm,
            0.032 - 2 / alphaterm
        ])
        beta = np.hstack([
            tau / betaterm,
            -5 * tau / (2 * betaterm),
            2 * tau / betaterm,
            -tau / (2 * betaterm)
        ])
        G, Fdiag = DW_flow(pD, pR, Viscos, cc, qk, alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)
    elif headloss == 'H-W':
        G = ResCoeff[:, np.newaxis] * abs(qk) ** (n_exp - 1)

    # calculate initial error
    err1 = np.linalg.norm(
        np.vstack((G * qk + A12 @ hk + A10 @ h0[:, np.newaxis] + A13 @ eta, A21 @ qk - d[:, np.newaxis])), ord=np.inf)

    nc = Z.shape[1]

    # nnz_ZZ = (Z.T @ sp.eye(NP) @ Z).count_nonzero()
    ZT = Z.T

    # Fdiag_old = sp.csc_matrix((NP, 1))
    # X = sp.csc_matrix((nc, nc))

    updates = np.arange(NP)
    n_upd = len(updates)

    for kk in range(max_iter):
        if headloss == 'H-W':
            Fdiag = n_exp * G
        sigma_max = np.max(Fdiag)
        t_k = np.maximum((sigma_max / kappa) - Fdiag, 0)
        Fdiag = Fdiag + t_k

        X = ZT @ sp.spdiags(Fdiag.T, [0], n_upd, n_upd) @ Z

        b = ZT @ ((Fdiag - G) * qk - A10 @ h0[:, np.newaxis] - A13 @ eta - Fdiag * x[:, np.newaxis])

        v = spla.spsolve(X, b)

        q = x[:, np.newaxis] + Z @ v[:, np.newaxis]

        b = A21 @ ((Fdiag - G) * qk - A10 @ h0[:, np.newaxis] - A13 @ eta - Fdiag * q)
        y = spla.spsolve(L_A12, P_r @ b)
        h = P_r.T * spla.spsolve(L_A12.T, y)

        if headloss == 'D-W':
            G, Fdiag = DW_flow(pD, pR, Viscos, cc, q, alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)
        elif headloss == 'H-W':
            G[updates] = ResCoeff[updates, np.newaxis] * abs(q[updates]) ** (n_exp - 1)

        if Method == 'fix_headTarget':
            h_ideal = h
            h_ideal[TargetNodes] = h_target
            if headloss == 'H-W':
                eta = -A12[valves, :] @ h_ideal - A10[valves, :] @ h0 - q[valves] * ResCoeff[valves, :] * abs(
                    q[valves]) ** (n_exp - 1)
            elif headloss == 'D-W':
                eta = -A12[valves, :] @ h_ideal - A10[valves, :] @ h0 - G[valves] * q[valves]

        err1 = np.linalg.norm(np.vstack(
            (G * q + A12 @ h[:, np.newaxis] + A10 @ h0[:, np.newaxis] + A13 @ eta, A21 @ q - d[:, np.newaxis])),
                              ord=np.inf)

        ERRORS = err1

        if err1 < tol:
            break
        else:
            qk = q

    if len(CL) > 0:
        qorig = np.zeros((len(CL) + NP, 1))
        s = set(CL)
        qorig[[x for x in np.arange(len(CL) + NP) if x not in s]] = q
        q = qorig

    check = int(ERRORS > tol)

    return q.flatten(), h, err1, kk, CONDS, check


# def DW_temp(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, NP, nulldata, auxdata):
#
#     rho = 998.2 # density of water(kg / m ^ 3) at 20 degrees C
#     mu = 1.002e-3 # dynamic viscosity(kg / m / s) at 20 degrees C
#     Viscos = mu / rho # kinematic viscosity(m ^ 2 / s) % viscosity of water at 20 deg C in m ^ 2 / s
#     pD = auxdata['D'] # Pipe Diameters
#     pR = auxdata['C']  # Pipe Rougness
#     pL = auxdata['L']  # ResCoeff = data.ResCoeff
#     CL = auxdata['closed_pipes']
#
#
#     if len(CL) > 0:
#         CLmask = np.ones(NP, dtype=bool)
#         CLmask[CL] = False
#         A12 = A12[CLmask, :]
#         A21 = A12.T
#         A10 = A10[CLmask, :]
#         NP = NP - len(CL)
#         A13 = A13[CLmask, :]
#         # ResCoeff = ResCoeff[CLmask, :]
#         qk = qk[CLmask, :]
#         pD = pD[CLmask, :]
#         pR = pR[CLmask, :]
#         pL = pL[CLmask, :]
#
#     x = nulldata['x']
#     Z = nulldata['Z']
#     P_r = nulldata['Pr']
#     L_A12 = nulldata['L_A12']
#
#     CONDS = 1
#     ERRORS = 1
#
#     max_iter = auxdata['max_iter']
#     tol = auxdata['tol_err']
#     kappa = auxdata['kappa']
#
#     Method = 'standard'
#     if len(A13) == 0:
#         eta = np.zeros((1, 1))
#         A13 = sp.csc_matrix((NP, 1))
#     elif len(eta) == 0 and 'TargetNodes' in auxdata:
#         Method = 'fix_headTarget'
#         TargetNodes = auxdata['TargetNodes']
#         h_target = auxdata['h_target']
#         valves = auxdata['valves']
#         eta = np.zeros((len(TargetNodes), 1))
#
#     cc = 5.74 * (np.pi * Viscos / 4) ** 0.9
#     ResCoeff_LAMIflow = (128 * Viscos / (np.pi * 9.81)) * pL * (pD ** -4)
#     K = pL / (pD ** 5)
#
#     # Rek = np.abs(qk / (np.pi * pD / 4)) / Viscos
#     # p_etak = Rek / 2000
#     # thetak = ((1 / 3.7) * pR) / pD + cc * np.abs(pD / qk) ** 0.9
#
#     ResCoeff = np.zeros((NP, 1))
#     G = np.zeros((NP, 1))
#     Fdiag = np.ones((NP, 1))
#
#     # cubic interpolating spline
#     theta_hat = ((1 / 3.7) * pR) / pD + (5.74 / 4000 ** 0.9)
#     tau = 0.00514215
#     xi = -0.86859
#     alphaterm = xi ** 2 * np.log(theta_hat) ** 2
#     betaterm = xi ** 3 * np.log(theta_hat) ** 3
#     alpha = np.hstack([
#         5 / alphaterm,
#         0.128 - 12 / alphaterm,
#         -0.128 + 9 / alphaterm,
#         0.032 - 2 / alphaterm
#     ])
#     beta = np.hstack([
#         tau / betaterm,
#         -5 * tau / (2 * betaterm),
#         2 * tau / betaterm,
#         -tau / (2 * betaterm)
#     ])
#
#     # LAMIflow = np.ravel(Rek <= 2000)
#     # TRANflow = np.ravel((Rek > 2000) & (Rek < 4000))
#     # TURBflow = np.ravel(Rek >= 4000)
#     # f = np.zeros(np.sum(TRANflow))
#     # ff = np.zeros(np.sum(TRANflow))
#     # for i in range(4):
#     #     f = f + ((alpha[TRANflow, i] + (beta[TRANflow, i] / thetak[TRANflow, 0])) * p_etak[TRANflow, 0] ** i)
#     #     ff = ff + ((9 * cc / 10 * beta[TRANflow, i] / thetak[TRANflow, 0] ** 2. * np.abs(
#     #         pD[TRANflow, 0] / qk[TRANflow, 0]) ** 0.9 + (2 + i) * (
#     #         alpha[TRANflow, i] + beta[TRANflow, i] / thetak[TRANflow, 0])) * p_etak[TRANflow, 0] ** i)
#     #
#     #
#     # ResCoeff[LAMIflow] = ResCoeff_LAMIflow[LAMIflow] # (128 * Viscos / (pi * 9.81)). * pL(LAMIflow). * (pD(LAMIflow). ^ -4)
#     # ResCoeff[TRANflow,0] = (f * (8 / (np.pi ** 2 * 9.81))) * K[TRANflow,0] # pL[TRANflow]. * (pD[TRANflow]. ^ -5);
#     # ResCoeff[TURBflow,0] = (1 / (np.log(thetak[TURBflow,0]) ** 2)) * ((2 * np.log(10) ** 2) / (np.pi ** 2 * 9.81)) * K[TURBflow,0] #.*pL(TURBflow). * (pD(TURBflow). ^ -5)
#     # G[LAMIflow] = ResCoeff[LAMIflow]
#     # G[TRANflow | TURBflow]=ResCoeff[TRANflow | TURBflow]*np.abs(qk[TRANflow | TURBflow]) #.^ (n_exp( or (TRANflow, TURBflow)) - 1)
#
#     G, Fdiag = DW_flow(pD, pR, Viscos, cc, qk, alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)
#
#     err1 = np.linalg.norm(
#         np.vstack((G * qk + A12 @ hk + A10 @ h0[:, np.newaxis] + A13 @ eta, A21 @ qk - d[:, np.newaxis])), ord=np.inf)
#
#     nc = Z.shape[1]
#
#     nnz_ZZ = (Z.T @ sp.eye(NP) @ Z).count_nonzero()
#     ZT = Z.T
#
#     Fdiag_old = sp.csc_matrix((NP, 1))
#     X = sp.csc_matrix((nc, nc))
#
#     updates = np.arange(NP)
#     n_upd = len(updates)
#
#     q = qk
#     for kk in range(max_iter):
#         # Fdiag = n_exp * G # HW
#
#         # if kk == 0:
#         #     q = qk
#         #     theta = thetak
#         #     Re = Rek
#         #     p_eta = p_etak
#
#         # Fdiag[LAMIflow] = G[LAMIflow]
#         # Fdiag[TRANflow,0] = (ff * (8 / (np.pi ** 2 * 9.81))) * K[TRANflow,0] * np.abs(q[TRANflow,0])
#         # Fdiag[TURBflow, 0] = ((2 * np.log(10) ** 2) / (np.pi ** 2 * 9.81)) * K[TURBflow, 0] * np.abs(q[TURBflow, 0]) * (
#         #     np.log(theta[TURBflow, 0]) ** -2) * (2 + 9 * cc / (5 * theta[TURBflow, 0] * np.log(
#         #     theta[TURBflow, 0])) * np.abs(pD[TURBflow, 0] / q[TURBflow, 0]) ** 0.9)
#
#
#
#         sigma_max = np.max(Fdiag)
#         t_k = np.maximum((sigma_max / kappa) - Fdiag, 0)
#         Fdiag = Fdiag + t_k
#
#         X = ZT @ sp.spdiags(Fdiag.T, [0], n_upd, n_upd) @ Z
#
#         b = ZT @ ((Fdiag - G) * qk - A10 @ h0[:, np.newaxis] - A13 @ eta - Fdiag * x[:, np.newaxis])
#
#         v = spla.spsolve(X, b)
#
#         q = x[:, np.newaxis] + Z @ v[:, np.newaxis]
#
#         b = A21 @ ((Fdiag - G) * qk - A10 @ h0[:, np.newaxis] - A13 @ eta - Fdiag * q)
#         y = spla.spsolve(L_A12, P_r @ b)
#         h = P_r.T * spla.spsolve(L_A12.T, y)
#
#         G, Fdiag = DW_flow(pD, pR, Viscos, cc, q, alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)
#
#         # Re = np.abs(q / (np.pi * pD / 4)) / Viscos
#         # p_eta = Re / 2000
#         # theta = ((1 / 3.7) * pR) / pD + cc * np.abs(pD / q) ** 0.9
#         #
#         # LAMIflow = np.ravel(Re <= 2000)
#         # TRANflow = np.ravel((Re > 2000) & (Re < 4000))
#         # TURBflow = np.ravel(Re >= 4000)
#         # f = np.zeros(np.sum(TRANflow))
#         # ff = np.zeros(np.sum(TRANflow))
#         # for i in range(4):
#         #     f = f + ((alpha[TRANflow, i] + (beta[TRANflow, i] / theta[TRANflow, 0])) * p_eta[TRANflow, 0] ** i)
#         #     ff = ff + ((9 * cc / 10 * beta[TRANflow, i] / theta[TRANflow, 0] ** 2. * np.abs(
#         #         pD[TRANflow, 0] / q[TRANflow, 0]) ** 0.9 + (2 + i) * (
#         #         alpha[TRANflow, i] + beta[TRANflow, i] / theta[TRANflow, 0])) * p_eta[TRANflow, 0] ** i)
#         #
#         # G[LAMIflow] = ResCoeff_LAMIflow[LAMIflow] # (128 * Viscos / (pi * 9.81)). * pL(LAMIflow). * (pD(LAMIflow). ^ -4);
#         # G[TRANflow,0] = (f * (8 / (np.pi ** 2 * 9.81))) * K[TRANflow,0] * np.abs(q[TRANflow,0]) #.^ (n_exp[TRANflow] - 1);
#         # G[TURBflow,0] = (1 / (np.log(theta[TURBflow,0]) ** 2)) * ((2 * np.log(10) ** 2) / (np.pi ** 2 * 9.81)) * K[TURBflow,0] * np.abs(
#         #     q[TURBflow,0]) #.^ (n_exp(TURBflow) - 1);
#
#
#         if Method == 'fix_headTarget':
#             h_ideal = h
#             h_ideal[TargetNodes] = h_target
#             # eta = -A12[valves, :] @ h_ideal - A10[valves, :] @ h0 - q[valves] * ResCoeff[valves, :] * abs(
#             #     q[valves]) ** (n_exp - 1)
#
#             eta = -A12[valves, :] @ h_ideal - A10[valves, :] @ h0 - G[valves] * q[valves]
#
#         # G[updates] = ResCoeff[updates, np.newaxis] * abs(q[updates]) ** (n_exp - 1)
#         err1 = np.linalg.norm(np.vstack(
#             (G * q + A12 @ h[:, np.newaxis] + A10 @ h0[:, np.newaxis] + A13 @ eta, A21 @ q - d[:, np.newaxis])),
#             ord=np.inf)
#
#         ERRORS = err1
#
#         if err1 < tol:
#             break
#         else:
#             qk = q
#
#     if len(CL) > 0:
#         qorig = np.zeros((len(CL) + NP, 1))
#         s = set(CL)
#         qorig[[x for x in np.arange(len(CL) + NP) if x not in s]] = q
#         q = qorig
#
#     check = int(ERRORS > tol)
#
#     return q.flatten(), h, err1, kk, CONDS, check


def DW_flow(pD,pR,Viscos,cc,q,alpha,beta,ResCoeff_LAMIflow,K,G,Fdiag):
    Re = np.abs(q / (np.pi * pD / 4)) / Viscos
    p_eta = Re / 2000
    theta = ((1 / 3.7) * pR) / pD + cc * np.abs(pD / q) ** 0.9

    LAMIflow = np.ravel(Re <= 2000)
    TRANflow = np.ravel((Re > 2000) & (Re < 4000))
    TURBflow = np.ravel(Re >= 4000)
    f = np.zeros(np.sum(TRANflow))
    ff = np.zeros(np.sum(TRANflow))
    for i in range(4):
        f = f + ((alpha[TRANflow, i] + (beta[TRANflow, i] / theta[TRANflow, 0])) * p_eta[TRANflow, 0] ** i)
        ff = ff + ((9 * cc / 10 * beta[TRANflow, i] / theta[TRANflow, 0] ** 2. * np.abs(
            pD[TRANflow, 0] / q[TRANflow, 0]) ** 0.9 + (2 + i) * (
                            alpha[TRANflow, i] + beta[TRANflow, i] / theta[TRANflow, 0])) * p_eta[TRANflow, 0] ** i)

    G[LAMIflow] = ResCoeff_LAMIflow[LAMIflow]  # (128 * Viscos / (pi * 9.81)). * pL(LAMIflow). * (pD(LAMIflow). ^ -4);
    G[TRANflow, 0] = (f * (8 / (np.pi ** 2 * 9.81))) * K[TRANflow, 0] * np.abs(
        q[TRANflow, 0])  # .^ (n_exp[TRANflow] - 1);
    G[TURBflow, 0] = (1 / (np.log(theta[TURBflow, 0]) ** 2)) * ((2 * np.log(10) ** 2) / (np.pi ** 2 * 9.81)) * K[
        TURBflow, 0] * np.abs(
        q[TURBflow, 0])  # .^ (n_exp(TURBflow) - 1);

    Fdiag[LAMIflow] = G[LAMIflow]
    Fdiag[TRANflow, 0] = (ff * (8 / (np.pi ** 2 * 9.81))) * K[TRANflow, 0] * np.abs(q[TRANflow, 0])
    Fdiag[TURBflow, 0] = ((2 * np.log(10) ** 2) / (np.pi ** 2 * 9.81)) * K[TURBflow, 0] * np.abs(q[TURBflow, 0]) * (
            np.log(theta[TURBflow, 0]) ** -2) * (2 + 9 * cc / (5 * theta[TURBflow, 0] * np.log(
        theta[TURBflow, 0])) * np.abs(pD[TURBflow, 0] / q[TURBflow, 0]) ** 0.9)

    return G, Fdiag




if __name__ == '__main__':
    import os
    import pyWDN
    import time

    #### test D-W solver ######

    filename = 'BURWELMA_2019-12-27_-_2019-12-28'
    temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    temp.evaluate_hydraulics(H0=temp.H0_extended,demands=temp.demands_extended/1000,C=temp.C/1000)
    print(temp.q_sim)

    #### test H-W solver ######

    # filename = '25nodesData'
    # temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    # temp.evaluate_hydraulics()
    # print(temp.q_sim)


    # #### test julia times vs python #####
    # filename = '25nodesData'
    # temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    # for _ in range(5):
    #     t = time.time()
    #     temp.evaluate_hydraulics()
    #     elapsed = time.time() - t
    #     print('Python: ' + str(elapsed))  # MATLAB runs the same network simulation in 0.155s
    #
    # from pyWDN.Solvers.call_ju import *
    #
    # # call_julia('hydraulics_ju.jl', 'funfun', temp.A12)
    # for _ in range(5):
    #     t = time.time()
    #     H, Q = call_julia('hydraulics_ju.jl', 'evaluate_hydraulics', temp.A12, temp.A10, temp.C, temp.D, temp.demands,
    #                       temp.H0, temp.IndexValves, temp.L, temp.nn, temp.np, temp.nl, temp.headloss, temp.nulldata,
    #                       temp.auxdata)
    #     elapsed2 = time.time() - t
    #     print('Julia: ' + str(elapsed2))


# todo: if ray is compatable on windows in the future, attempt to implement the hydraulic solver by using a pandas approach instead and parallel computing (modin package)
# todo: add some sort of auto units checker / converter

