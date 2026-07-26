# Loads the frozen CNN weights and makes predict_solver(A) and predict_solver_probs(A)
using Flux: Flux
using SparseArrays: SparseMatrixCSC

# Path to the bundled model weights (relative to this file)
const _MODEL_PATH = joinpath(@__DIR__, "..", "artifacts", "solver_model_cnn.jld2")

# Module-level model cache, loaded once on first call
const _MODEL = Ref{Any}(nothing)
const _LABELS = Ref{Vector{String}}(String[])


function _load_model!()
    # Loads the bundled model on the first call. Subsequent calls use the cached model.
    return if _MODEL[] === nothing
        if !isfile(_MODEL_PATH)
            error(
                """
                Model weights not found at: $_MODEL_PATH

                Please ensure solver_model_cnn.jld2 is present in the
                artifacts/ directory of the NeuralLinearSolve package.
                """
            )
        end
        model_data = JLD2.load(_MODEL_PATH)
        _MODEL[] = model_data["model_cpu"]
        _LABELS[] = model_data["label_names"]
    end
end

"""
    predict_solver(A::SparseMatrixCSC) -> Symbol

Predict the linear solver expected to be fastest for a sparse matrix.

The bundled convolutional neural network converts `A` into a fixed-size sparsity
image and selects one of the supported solver labels. The model is loaded on the
first call and then cached for subsequent predictions.

# Arguments

- `A`: Sparse matrix whose sparsity pattern is used for the prediction.

# Returns

- `Symbol`: A predicted solver label, currently one of `:UMFPACK`, `:KLU`, or
  `:Pardiso`.

# Examples

```jldoctest
julia> using NeuralLinearSolve, SparseArrays

julia> A = sprand(16, 16, 0.2);

julia> predict_solver(A) in (:UMFPACK, :KLU, :Pardiso)
true
```
"""
function predict_solver(A::SparseMatrixCSC)
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
    predict_solver_probs(A::SparseMatrixCSC) -> Dict{Symbol, Float32}

Predict the probability assigned to each supported linear solver for a sparse matrix.

The probabilities are produced by the same cached model used by [`predict_solver`](@ref)
and are useful when downstream code needs to choose among close predictions.

# Arguments

- `A`: Sparse matrix whose sparsity pattern is used for the prediction.

# Returns

- `Dict{Symbol, Float32}`: A mapping from each supported solver label to its predicted
  probability.

# Examples

```jldoctest
julia> using NeuralLinearSolve, SparseArrays

julia> A = sprand(16, 16, 0.2);

julia> probs = predict_solver_probs(A);

julia> isapprox(sum(values(probs)), 1f0)
true
```
"""
function predict_solver_probs(A::SparseMatrixCSC)
    _load_model!()

    X = matrix_to_spy(A)
    model = _MODEL[]
    labels = _LABELS[]

    Flux.testmode!(model)
    probs = vec(model(X))

    return Dict(Symbol(labels[i]) => probs[i] for i in 1:length(labels))
end
