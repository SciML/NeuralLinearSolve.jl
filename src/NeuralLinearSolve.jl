module NeuralLinearSolve

using SparseArrays
using Flux
using BSON
using Images
using FileIO

include("spy_plot.jl")
include("predict.jl")

export predict_solver, predict_solver_probs

end
