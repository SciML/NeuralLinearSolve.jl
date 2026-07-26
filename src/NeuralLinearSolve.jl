module NeuralLinearSolve

using SparseArrays: SparseArrays
using Flux: Flux
using Images: Images
using JLD2: JLD2
using NNlib: NNlib

include("spy_plot.jl")
include("predict.jl")

export predict_solver, predict_solver_probs

end
