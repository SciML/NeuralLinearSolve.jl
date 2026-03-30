# Loads the frozen CNN weights and makes predict_solver(A) and predict_solver_probs(A)
using BSON: @load
using Flux
using SparseArrays

# Path to the bundled model weights (relative to this file)
const _MODEL_PATH = joinpath(@__DIR__, "..", "artifacts", "solver_model_cnn.bson")

# Module-level model cache, loaded once on first call
const _MODEL = Ref{Any}(nothing)
const _LABELS = Ref{Vector{String}}(String[])


function _load_model!()
    # Loads CNN model from the bundled BSON file on first call. Subsequent calls are no-ops (model is cached in `_MODEL`).
    return if _MODEL[] === nothing
        if !isfile(_MODEL_PATH)
            error(
                """
                Model weights not found at: $_MODEL_PATH

                Please ensure solver_model_cnn.bson is present in the
                artifacts/ directory of the NeuralLinearSolve package.
                """
            )
        end
        @load _MODEL_PATH model_cpu label_names
        _MODEL[] = model_cpu
        _LABELS[] = label_names
    end
end

"""
# Example
```julia
using SparseArrays, NeuralLinearSolve

A = sprand(1000, 1000, 0.01)
solver = predict_solver(A)  # :KLU
```
"""
function predict_solver(A::SparseMatrixCSC)
    # Input: sparse matrix A
    # Output: Predicted faster linear solver for A (one of `:UMFPACK`, `:KLU`, `:Pardiso)
    _load_model!()

    # Generate SPY plot and run CNN forward pass
    X = matrix_to_spy(A)
    model = _MODEL[]
    labels = _LABELS[]

    # Run in eval mode (disables Dropout)
    Flux.testmode!(model)
    probs = model(X)
    idx = argmax(vec(probs))

    return Symbol(labels[idx])
end

"""
# Example
```julia
probs = predict_solver_probs(A)
# Dict(:UMFPACK => 0.05, :KLU => 0.91, :Pardiso => 0.04)
```
"""
function predict_solver_probs(A::SparseMatrixCSC)
    # Input: sparse matrix A
    # Output: Predicted probability for each solver (for each of `:UMFPACK`, `:KLU`, `:Pardiso)
    _load_model!()

    X = matrix_to_spy(A)
    model = _MODEL[]
    labels = _LABELS[]

    Flux.testmode!(model)
    probs = vec(model(X))

    return Dict(Symbol(labels[i]) => probs[i] for i in 1:length(labels))
end
