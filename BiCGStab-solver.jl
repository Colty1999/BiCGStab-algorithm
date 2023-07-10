using IncompleteLU, IterativeSolvers, Krylov
using SparseArrays, LinearOperators, LinearAlgebra

include("./problems/nonsymmetric-problem-fvm.jl")
include("./problems/symmetric-problem-fdm.jl")
include("./problems/symmetric-problem-fem.jl")

function fdm(n)
    A, b = fdmproblem(n)
    LU = IncompleteLU.ilu(A)
    itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)

    print("FDM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

function fem(n)
    A, b = femproblem(n)
    LU = IncompleteLU.ilu(A)
    itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)

    print("FEM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

function fvm()
    f = zeros(1000)
    f[200:300] .= 1.0
    A, b = fvmproblem(f)
    LU = IncompleteLU.ilu(A)
    itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)

    print("FVM\n")
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

n = 10
fdm(n)
fem(n)
fvm()