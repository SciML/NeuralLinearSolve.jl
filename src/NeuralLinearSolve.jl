module NeuralLinearSolve

using SparseArrays: SparseArrays
using Flux: Flux
using BSON: BSON
using Images: Images
using FileIO: FileIO

include("spy_plot.jl")
include("predict.jl")

export predict_solver, predict_solver_probs

end
