using IterativeSolvers, Krylov
using SparseArrays, LinearOperators, LinearAlgebra

include("./problems/nonsymmetric-problem-fvm.jl")
include("./problems/symmetric-problem-fdm.jl")
include("./problems/symmetric-problem-fem.jl")

function fdm(n)
    A, b = fdmproblem(n)
    itSol = IterativeSolvers.bicgstabl(A, b)
    krl, stats = Krylov.bicgstab(A, b, history=true)

    print("FDM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

function fem(n)
    A, b = femproblem(n)
    itSol = IterativeSolvers.bicgstabl(A, b)
    krl, stats = Krylov.bicgstab(A, b, history=true)

    print("FEM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

function fvm()
    f = zeros(1000)
    f[200:300] .= 1.0
    A, b = fvmproblem(f)
    itSol = IterativeSolvers.bicgstabl(A, b)
    krl, stats = Krylov.bicgstab(A, b, history=true)

    print("FVM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

n = 10
fdm(n)
fem(n)
fvm()