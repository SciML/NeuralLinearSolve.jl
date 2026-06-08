# predict.jl
# Loads the frozen CNN weights and exposes predict_solver(A).
# Regression version: model outputs [log(t_umfpack/t_klu), log(t_pardiso/t_klu)]
# Prediction = argmin of [log_umf_klu, 0, log_pard_klu] (KLU = 0 baseline)

using BSON: @load
using Flux
using SparseArrays


const _MODEL_PATH = joinpath(@__DIR__, "..", "artifacts", "solver_model_cnn.bson")
const _MODEL  = Ref{Any}(nothing)
const _LABELS = Ref{Vector{String}}(String[])

"""
    _load_model!()

Loads the CNN model from the BSON file on first call. Subsequent calls are from cache.
"""
function _load_model!()
    if _MODEL[] === nothing
        if !isfile(_MODEL_PATH)
            error("""
            Model weights not found at: $_MODEL_PATH

            Please ensure solver_model_cnn.bson is present in the
            artifacts/ directory of the NeuralLinearSolve package.
            """)
        end
        @load _MODEL_PATH model_cpu label_names
        _MODEL[]  = model_cpu
        _LABELS[] = label_names
    end
end

"""
    predict_solver(A::SparseMatrixCSC) -> Symbol

Predicts the fastest direct linear solver for the sparse matrix `A`n using a pretrained CNN on spy plots.
Returns one of: `:UMFPACK`, `:KLU`, `:Pardiso`

# Example
```julia
using SparseArrays, NeuralLinearSolve

A = sprand(1000, 1000, 0.01)
solver = predict_solver(A)  # e.g. :KLU
```
"""
function predict_solver(A::SparseMatrixCSC)
    _load_model!()

    # Warn if complex
    if eltype(A) <: Complex
        @warn "predict_solver was trained on real-valued matrices only. Predictions may be unreliable for complex matrices."
    end

    X      = matrix_to_spy(A)
    model  = _MODEL[]
    labels = _LABELS[]

    Flux.testmode!(model)
    out = vec(model(X))   # [log(t_umf/t_klu), log(t_pard/t_klu)]

    # KLU is baseline (score = 0)
    # UMFPACK score = log(t_umf/t_klu): positive = UMFPACK slower than KLU
    # Pardiso score = log(t_pard/t_klu): positive = Pardiso slower than KLU
    scores = [out[1], 0.0f0, out[2]]
    idx    = argmin(scores)   # 1=UMFPACK, 2=KLU, 3=Pardiso

    return Symbol(labels[idx])
end

"""
    predict_solver_scores(A::SparseMatrixCSC) -> Dict{Symbol, Float32}

Return a proxy score for each solver derived from the predicted log timing ratios. Lower score = faster predicted solver. Scores are shifted so the
minimum is zero and negated so that higher = more confident recommendation.

Note: These are not probabilities in [0,1] but relative speed scores useful for ranking solvers.

# Example
```julia
scores = predict_solver_scores(A)
# Dict(:UMFPACK => 0.0, :KLU => 1.2, :Pardiso => 0.3)
# Higher score = more strongly recommended
```
"""
function predict_solver_scores(A::SparseMatrixCSC)
    _load_model!()

    if eltype(A) <: Complex
        @warn "predict_solver was trained on real-valued matrices only. Predictions may be unreliable for complex matrices."
    end

    X      = matrix_to_spy(A)
    model  = _MODEL[]
    labels = _LABELS[]

    Flux.testmode!(model)
    out = vec(model(X))   # [log(t_umf/t_klu), log(t_pard/t_klu)]

    # scores: lower = faster
    scores = [out[1], 0.0f0, out[2]]

    # shift so min = 0, then negate so higher = better
    shifted = scores .- minimum(scores)
    ranking = maximum(shifted) .- shifted

    return Dict(Symbol(labels[i]) => ranking[i] for i in 1:length(labels))
end
