using IncompleteLU, IterativeSolvers, Krylov
using SparseArrays, LinearOperators, LinearAlgebra

include("./Problems/nonsymmetric-problem-fvm.jl")
include("./Problems/symmetric-problem-fdm.jl")
include("./Problems/symmetric-problem-fem.jl")
include("./Components/BiCGStab-custom.jl")
include("./Components/Incomplete-LU-custom.jl")

function fdm(n)
    print("====================================\n")
    print("FDM\n")
    A, b = fdmproblem(n)
    LU = IncompleteLU.ilu(A)
    my_LU = custom_iLU(A)

    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, my_LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")

end

function fem(n)
    print("====================================\n")
    print("FEM\n")
    A, b = femproblem(n)
    LU = IncompleteLU.ilu(A)
    my_LU = custom_iLU(A)


    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, my_LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

function fvm()
    f = zeros(1000)
    f[200:300] .= 1.0
    print("====================================\n")
    print("FVM\n")
    A, b = fvmproblem(f)
    LU = IncompleteLU.ilu(A)
    my_LU = custom_iLU(A)


    x = zeros(size(b))
    @time bcg_wyn = custom_BICGStab(A, b, my_LU, x, 1.0e-8)
    print(" [Left preconditioning] custom_BICGStab Residual norm: ", norm(b - A * bcg_wyn), "\n")


    @time itSol = IterativeSolvers.bicgstabl(A, b, Pl=LU)
    print(" [Left preconditioning] IterativeSolvers Residual norm: ", norm(b - A * itSol), "\n")

    n = size(b, 1)
    opP = LinearOperator(Float64, n, n, false, false, (y, v) -> ldiv!(y, LU, v))
    @time krl, stats = Krylov.bicgstab(A, b, history=true, M=opP)
    print(" [Left preconditioning] Krylov residual norm: ", norm(b - A * krl), "\n\n")
end

n = 20
fdm(n)
fem(n)
fvm()