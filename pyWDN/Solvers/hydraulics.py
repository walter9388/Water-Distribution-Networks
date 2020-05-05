import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import psutil
from datetime import datetime as dt
from datetime import timedelta as td
from tqdm import tqdm


def get_scaled_demand(demand, flow_in, flow_dt, flow_out=None):
    """
    :param demand: 2D-numpy-array of dimensions nn x nl
    :param flow_in: 2D-numpy-array of dimensions n_flows_in x nl
    :param flow_dt: list of datetimes corresponding to n_flows (currently 15 min timesteps only)
    :param flow_out: OPTIONAL: None if no measured out flow avaliable, else a list of dictionaries which the outflow
    node ID (so the demand at that point can be adjusted to mimic flow out) and a 1D-numpy-array of dimension nl,
    e.g. flow_out = [dict(index=7, data=np.array([1,2,3,4]))]
    :return: new_demands: same dimensions as old demands
    """

    if (len(flow_dt) != len(flow_in)):
        raise ValueError('flow and datetime should be the same length')

    # determine the total amount of water used at each time step (i.e. flow_in - flow_out)
    if flow_out is not None:
        total_dem = flow_in.sum(axis=1) - np.vstack([flow_out[ii]['data'] for ii in range(len(flow_out))]).sum(axis=0)
    else:
        total_dem = flow_in.sum(axis=1)

    # create multiplier array aligned with datetime for slicing (written cumbersomely, can be updated in the future)
    int_secs = (60*60*24)/demand.shape[1]
    mult_dt = np.arange(dt(2020, 1, 1), dt(2020, 1, 2), td(seconds=int_secs), dtype=dt)
    mult_dt = [(mult_dt[i].hour, mult_dt[i].minute) for i in range(len(mult_dt))]
    flow_dt = [(flow_dt[i].hour, flow_dt[i].minute) for i in range(len(flow_dt))]

    # get the index of which column in the demand series should be reallocated as a comparison of the datetime arrays
    demand_ts = [np.where((flow_dt[i] == np.array(mult_dt)).all(axis=1))[0][0] for i in range(len(flow_dt))]

    # create new rescaled demands based on the total demand data
    demand_sum = demand.sum(axis=0)
    new_demands = np.zeros((demand.shape[0], len(flow_dt)))
    for i in range(len(flow_dt)):
        new_demands[:, i] = demand[:, demand_ts[i]]*total_dem[i]/demand_sum[demand_ts[i]]

    # add any flows out as demands, thus addressing the mass balance
    if flow_out is not None:
        for ii in range(len(flow_out)):
            new_demands[flow_out[ii]['index'], :] += flow_out[ii]['data']

    return new_demands


def update_eta_manually(PRVs, inlet_pressure, outlet_pressure, flow, A12, A10, elev, H0, D, C, L, headloss, IndexValves):
    """
    :param PRV_links: list of link numbers
    :param inlet_pressure: 2D-numpy-array of dimensions nPRVs x nl
    :param outlet_pressure: 2D-numpy-array of dimensions nPRVs x nl
    :param flow: 2D-numpy-array of dimensions nPRVs x nl
    :return:
    """

    if (inlet_pressure.shape != outlet_pressure.shape) & (inlet_pressure.shape != flow.shape):
        raise ValueError('inlet pressure, outlet pressure and flow must be the same shape')
    elif np.any(np.isnan(inlet_pressure)) | np.any(np.isnan(outlet_pressure)) | np.any(np.isnan(flow)):
        raise ValueError('There cannot be any nan values in the pressure or flow arrays')

    nl = len(D)
    p_data = np.empty((A12.shape[1], nl)) * np.nan
    q_data = np.empty((A12.shape[0], nl)) * np.nan

    # add data into the blank arrays
    for ii in range(len(PRVs)):
        p_data[np.where(A12[1, :].toarray() == 1)[1][0], :] = inlet_pressure[ii, :]
        p_data[np.where(A12[1, :].toarray() == -1)[1][0], :] = outlet_pressure[ii, :]
        q_data[PRVs[ii], :] = flow[ii, :]

    eta, A13 = update_eta(PRVs, p_data, q_data, A12, A10, elev, nl, H0, D, C, L, headloss, IndexValves, [], [])
    return eta, A13



def update_eta(PRVs, p_data, q_data, A12, A10, elev, nl, H0, D, C, L, headloss, IndexValves, a, b):

    eta = np.zeros((len(PRVs), nl))
    A13 = np.zeros((q_data.shape[0], len(PRVs)))

    if len(PRVs) != 0:
        A13[PRVs, :] = np.eye(len(PRVs))
        if headloss['formula'] == 'QA':
            eta = -(A12[PRVs, :] @ (p_data + elev[:, np.newaxis] @ np.ones((1, nl))) + A10[PRVs, :] @ H0
                    + q_data[PRVs, :] * a[PRVs, :] * np.abs(q_data[PRVs,:]) + b[PRVs, :])
        elif headloss['formula'] == 'H-W':
            ResCoeff = 10.670 * L / (C ** headloss['n_exp'] * D ** 4.871)
            ResCoeff[IndexValves] = (8 / (np.pi ** 2 * 9.81)) * D[IndexValves] ** -4 * C[IndexValves]
            eta = -(A12[PRVs, :] @ (p_data + elev[:, np.newaxis] @ np.ones((1, nl))) + A10[PRVs, :] @ H0
                    + ResCoeff[PRVs, np.newaxis] * q_data[PRVs, :] * np.abs(q_data[PRVs, :]) ** (headloss['n_exp']-1))
        elif headloss['formula'] == 'D-W':
            Viscos = DW_water_constants()
            pD = D[PRVs, np.newaxis]
            pR = C[PRVs, np.newaxis]
            pL = L[PRVs, np.newaxis]

            cc, ResCoeff_LAMIflow, K = DW_constants(Viscos, pL, pD)

            ResCoeff = np.zeros((len(PRVs), 1))
            G = np.zeros((len(PRVs), 1))
            Fdiag = np.ones((len(PRVs), 1))

            alpha, beta = DW_cubic_spline(pR, pD)  # cubic interpolating spline

            G, _ = DW_flow(pD, pR, Viscos, cc, q_data[PRVs, :], alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)

            eta = -(A12[PRVs, :] @ (p_data + elev[:, np.newaxis] @ np.ones((1, nl)))
                    + A10[PRVs, :] @ H0 + G * q_data[PRVs, :])
        else:
            raise ValueError('headloss should be D-W, H-W, or QA')

    return eta, A13


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
        A13 = np.zeros((NP, 0))

    with tqdm(range(nl)) as pbar:
        for kk in pbar:
            pbar.set_description(f'Timestep {kk}')
            xtemp = spla.spsolve(nulldata['L_A12'], nulldata['Pr'] @ demands[:, kk])
            w = nulldata['Pr'].T @ spla.spsolve(nulldata['L_A12'].T, xtemp)
            xtemp = A12 @ w
            nulldata['x'] = xtemp
            q_0 = 0.3 * np.ones((NP, 1))
            h_0 = 130 * np.ones((nn, 1))
            if headloss['formula'] == 'H-W':
                auxdata['ResCoeff'] = 10.670 * L / (C ** headloss['n_exp'] * D ** 4.871)
                auxdata['ResCoeff'][IndexValves] = (8 / (np.pi ** 2 * 9.81)) * D[IndexValves] ** -4 * C[IndexValves]
            elif headloss['formula'] == 'D-W':
                auxdata['D'] = D[:, np.newaxis]
                auxdata['C'] = C[:, np.newaxis]
                auxdata['L'] = L[:, np.newaxis]
                auxdata['ResCoeff'] = np.zeros((NP, 1))
            Q[:, kk], H[:, kk], err_, iter_, CONDS[kk], check[kk] = solve_Nullspace(A12, A12.T, A10, A13,
                                                                                          H0[:, [kk]], Eta[:, kk],
                                                                                          headloss['n_exp'], q_0, h_0,
                                                                                          demands[:, kk], NP,
                                                                                          nulldata, auxdata,
                                                                                          headloss['formula'])
            err[kk], iter[kk] = err_, iter_

            pbar.set_postfix({'Error': err_, 'Iteration': iter_})
            # if print_timestep:
            #     print('Time Step: %i' % (kk + 1))
            # if ERRORS > auxdata['tol_err']:
            #     check[kk] = 1

        pbar.set_postfix_str('Done')

    return H, Q


def solve_Nullspace(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, NP, nulldata, auxdata, headloss):
    '''Based on the work of Edo Abraham, Imperial College London, 2014'''
    if headloss == 'D-W':
        Viscos = DW_water_constants()
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
    if A13.shape[1] == 0:
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
        cc, ResCoeff_LAMIflow, K = DW_constants(Viscos, pL, pD)

        ResCoeff = np.zeros((NP, 1))
        G = np.zeros((NP, 1))
        Fdiag = np.ones((NP, 1))

        alpha, beta = DW_cubic_spline(pR, pD)  # cubic interpolating spline
        G, Fdiag = DW_flow(pD, pR, Viscos, cc, qk, alpha, beta, ResCoeff_LAMIflow, K, G, Fdiag)
    elif headloss == 'H-W':
        G = ResCoeff[:, np.newaxis] * abs(qk) ** (n_exp - 1)

    # calculate initial error
    err1 = np.linalg.norm(
        np.vstack((G * qk + A12 @ hk + A10 @ h0 + A13 @ eta, A21 @ qk - d[:, np.newaxis])), ord=np.inf)

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

        b = ZT @ ((Fdiag - G) * qk - A10 @ h0 - A13 @ eta - Fdiag * x[:, np.newaxis])

        v = spla.spsolve(X, b)

        q = x[:, np.newaxis] + Z @ v[:, np.newaxis]

        b = A21 @ ((Fdiag - G) * qk - A10 @ h0 - A13 @ eta - Fdiag * q)
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
            (G * q + A12 @ h[:, np.newaxis] + A10 @ h0 + A13 @ eta, A21 @ q - d[:, np.newaxis])),
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


def DW_flow(pD,pR,Viscos,cc,q,alpha,beta,ResCoeff_LAMIflow,K,G,Fdiag):
    Re = np.abs(q / (np.pi * pD / 4)) / Viscos
    p_eta = Re / 2000
    q[q == 0] = 1e-12  # to prevent "RuntimeWarning: divide by zero encountered in true_divide" in next line
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


def DW_water_constants():
    rho = 998.2  # density of water(kg / m ^ 3) at 20 degrees C
    mu = 1.002e-3  # dynamic viscosity(kg / m / s) at 20 degrees C
    Viscos = mu / rho  # kinematic viscosity(m ^ 2 / s) % viscosity of water at 20 deg C in m ^ 2 / s
    return Viscos


def DW_constants(Viscos, pL, pD):
    cc = 5.74 * (np.pi * Viscos / 4) ** 0.9
    ResCoeff_LAMIflow = (128 * Viscos / (np.pi * 9.81)) * pL * (pD ** -4)
    K = pL / (pD ** 5)
    return cc, ResCoeff_LAMIflow, K


def DW_cubic_spline(epsilon, diameter):
    # cubic interpolating spline
    theta_hat = ((1 / 3.7) * epsilon) / diameter + (5.74 / 4000 ** 0.9)
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
    return alpha, beta


if __name__ == '__main__':
    import os
    import pyWDN
    import time

    #### test D-W solver ######

    filename = 'BURWELMA_2019-12-27_-_2019-12-28'
    temp = pyWDN.WDNbuild.BuildWDN_fromMATLABfile(filename)
    temp.evaluate_hydraulics(H0=temp.H0_extended, demands=temp.demands_extended/1000, C=temp.C/1000)
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

