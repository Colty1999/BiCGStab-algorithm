using IncompleteLU, IterativeSolvers, Krylov
using SparseArrays, LinearOperators, LinearAlgebra

include("./problems/nonsymmetric-problem-fvm.jl")
include("./problems/symmetric-problem-fdm.jl")
include("./problems/symmetric-problem-fem.jl")
include("./BiCGStab-custom.jl")

function fdm(n)
    print("FDM\n")
    A, b = fdmproblem(n)
    LU = IncompleteLU.ilu(A)


    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n")

end

function fem(n)
    print("FEM\n")
    A, b = femproblem(n)
    LU = IncompleteLU.ilu(A)


    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n")
end

function fvm()
    f = zeros(1000)
    f[200:300] .= 1.0
    print("FVM\n")
    A, b = fvmproblem(f)
    LU = IncompleteLU.ilu(A)


    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n")
end
n = 10
fdm(n)
fem(n)
fvm()