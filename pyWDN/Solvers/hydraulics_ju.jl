# module Test

using SparseArrays
using LinearAlgebra
using Printf
# using TimerOutputs
# to = TimerOutput()

# import BenchmarkTools: @btime
import Base: convert #, promote_rule, promote, @time

# PyObject to julia sparse conversion
mk_sparse(o) = SparseMatrixCSC(o.shape[1], o.shape[2], o.indptr .+ 1, o.indices .+ 1, o.data)
# converts PyObject to sparsematricies when needed
convert(::Type{SparseMatrixCSC}, o::PyObject) = mk_sparse(o)
# promotion ule doesn't work (not sure why), so i just convert throughout'
# promote_rule(::Type{SparseMatrixCSC}, ::Type{PyObject}) = SparseMatrixCSC

# function funfun(o::PyObject)
#     print(typeof(o),"\n")
# #     convert(SparseMatrixCSC,o)
# # #     @time o=mk_sparse(o)
# #     print(typeof(o),"\n")
#
#     x=mk_sparse(o)
#
#     print(x,"\n")
#     print(typeof(x),"\n")
#
# #     promote(x,o)
#
#     @btime $(x')*$(convert(SparseMatrixCSC,o))
#     @time a=(x')*(convert(SparseMatrixCSC,o))
#     print(a,"\n")
#     print(typeof(a),"\n")
# #     printting(o)
#     return o
# end

# function funfun(x)
#     print(typeof(x))
#     print(x)
#     return x
# end
#
# function printting(x)
#     print(x,"\n")
# end


function evaluate_hydraulics(A12,A10,C,D,demands,H0,IndexValves,L,nn,np,nl,headloss,nulldata,auxdata)
    H = fill!(zeros(nn,nl),NaN)
    Q = fill!(zeros(np,nl),NaN)
    err = fill!(zeros(nl),NaN)
    iter = fill!(zeros(nl),NaN)
    CONDS = fill!(zeros(nl),NaN)
    check = fill!(zeros(nl),NaN)

    # convert key matrices to sparse
    Pr=convert(SparseMatrixCSC,nulldata["Pr"])
    L_A12=convert(SparseMatrixCSC,nulldata["L_A12"])
    Z=convert(SparseMatrixCSC,nulldata["Z"])
    A12=convert(SparseMatrixCSC,A12)
    A10=convert(SparseMatrixCSC,A10)
    A13,Eta=("Eta" in keys(nulldata)
    ?
    (convert(SparseMatrixCSC,nulldata["A13"]),nulldata["Eta"])
    :
    (zeros(0,nl),zeros(0,nl)))

    start=time()
    for kk in 1:nl
        x = L_A12 \ ((Pr) * SparseVector(demands[:,kk]))
        w = convert(SparseMatrixCSC,Pr') * (convert(SparseMatrixCSC,L_A12') \ x)
        x = A12 * w
        q_0 = 0.3 * ones(np, 1)
        h_0 = 130 * ones(nn, 1)
        if headloss["formula"]=="H-W"
            ResCoeff = 10.670*L./(C.^headloss["n_exp"].*(D/1000).^4.871)
            ResCoeff[IndexValves] = ((8/(π^2*9.81)).*(D[IndexValves]/1000).^-4).*C[IndexValves]
            Q[:,kk], H[:,kk], err[kk], iter[kk], CONDS[kk], check[kk] = solveHW_Nullspace(A12, A12', A10, A13, H0[:,kk], Eta[:,kk], headloss["n_exp"], q_0, h_0, demands[:,kk], np, x, Z, Pr, L_A12, ResCoeff, auxdata)
        elseif headloss["formula"]=="D-W"
            print("you reyt love")
        end
#         @printf "Time Step: %i\n" kk
    end
    elapsed=time()-start
    print(elapsed)
#     display(to)
    return H, Q
end

function solveHW_Nullspace(A12, A21, A10, A13, h0, eta, n_exp, qk, hk, d, np, x, Z, P_r, L_A12, ResCoeff, auxdata)
    #= based on the work of Edo Abraham,
    Imperial College London,
    2014 =#

    CL = auxdata["closed_pipes"]
    if length(CL) > 0
        CLmask = trues(np)
        CLmask[CL] = false
        A12 = A12[CLmask, :]
        A21 = A12'
        A10 = A10[CLmask, :]
        np = np - length(CL)
        A13 = A13[CLmask, :]
        ResCoeff = ResCoeff[CLmask, :]
        qk = qk[CLmask, :]
    end

    CONDS = 1
    ERRORS = 1

    max_iter = auxdata["max_iter"]
    tol = auxdata["tol_err"]
    kappa = auxdata["kappa"]

    method = "standard"
    if length(A13)==0
        eta = 0
        A13=spzeros(np,1)
    elseif length(eta)==0 && "TargetNodes" in keys(auxdata)
        method = "fix_headTarget"
        TargetNodes = auxdata["TargetNodes"]
        h_target = auxdata["h_target"]
        valves = auxdata["valves"]
        eta = zeros(length(TargetNodes), 1)
    end


    G = ResCoeff.*abs.(qk).^(n_exp - 1)
    err1 = norm([G.*qk + A12*hk + A10*h0 + A13*eta ; A21*qk - d], Inf)

    nc = size(Z)[2]

    nnz_ZZ = nnz(Z'*I(np)*Z)

    Fdiag_old = spzeros(np, 1)
    X = spzeros(nc)

    updates=1:np
    n_upd = length(updates)

    ZT = Z'

    global h, kkk

    for kk in 1:max_iter
        Fdiag = n_exp * G
        sigma_max = maximum(Fdiag)
        t_k = max.((sigma_max / kappa).-Fdiag, 0)
        Fdiag = Fdiag + t_k

        X = ZT*spdiagm(n_upd,n_upd,0 => Fdiag[:])*Z

        b = ZT*((Fdiag-G).*qk-A10*h0-A13*eta-Fdiag.*x)

        v = X\b

        q = x + Z * v

        b = A21*((Fdiag-G).*qk-A10*h0-A13*eta-Fdiag.*q)
        y=L_A12\(P_r * b)

        h=P_r'*(L_A12'\y)
        if method == "fix_headTarget"
            h_ideal=h
            h_ideal[TargetNodes]=h_target
            eta=-A12[valves,:]*h_ideal - A10[valves,:]*h0 - q[valves].*ResCoeff[valves,:].*abs(q[valves]).^(n_exp - 1)
        end

        G[updates] = ResCoeff[updates].*abs.(q[updates]).^(n_exp - 1)
        err1 = norm([G.*q + A12*h + A10*h0 + A13*eta ; A21*qk - d], Inf)

        if err1 < tol
            kkk=kk
            ERRORS = err1
            break
        else
            qk = q
        end
    end

    if length(CL)>0
        qorig = zeros(length(CL) + np, 1)
        qorig[sort(Int64.(setdiff!(Set(1:(length(CL) + np)),Set(CL))))] = q
        q = qorig
    end

    check = ERRORS > tol

    return q,h,ERRORS,kkk,CONDS,check
end


# end
