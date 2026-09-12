using BenchmarkTools
using NeuralLinearSolve
using SparseArrays
using LinearAlgebra

const SUITE = BenchmarkGroup()

A_100 = sprand(100, 100, 0.05)
A_1000 = sprand(1000, 1000, 0.01)
A_diag = sparse(Diagonal(rand(500)))

SUITE["spy_plot"] = BenchmarkGroup()
SUITE["spy_plot"]["d100"] = @benchmarkable NeuralLinearSolve.matrix_to_spy($A_100)
SUITE["spy_plot"]["d1000"] = @benchmarkable NeuralLinearSolve.matrix_to_spy($A_1000)

SUITE["predict"] = BenchmarkGroup()
SUITE["predict"]["solver_d100"] = @benchmarkable predict_solver($A_100)
SUITE["predict"]["solver_d1000"] = @benchmarkable predict_solver($A_1000)
SUITE["predict"]["solver_diag"] = @benchmarkable predict_solver($A_diag)
SUITE["predict"]["probs_d100"] = @benchmarkable predict_solver_probs($A_100)
SUITE["predict"]["probs_d1000"] = @benchmarkable predict_solver_probs($A_1000)
