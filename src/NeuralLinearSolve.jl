module NeuralLinearSolve

using SparseArrays: SparseArrays
using Flux: Flux
using Images: Images
using JLD2: JLD2
using NNlib: NNlib
using PrecompileTools: @compile_workload, @setup_workload

include("spy_plot.jl")
include("predict.jl")

export predict_solver, predict_solver_probs

@setup_workload begin
    @compile_workload begin
        A = SparseArrays.sparse([1, 2, 3], [1, 2, 3], [1.0, 2.0, 3.0], 3, 3)
        predict_solver(A)
        predict_solver_probs(A)
    end
end

end
