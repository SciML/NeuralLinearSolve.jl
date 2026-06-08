# NeuralLinearSolve

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://artemispados.github.io/NeuralLinearSolve.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://artemispados.github.io/NeuralLinearSolve.jl/dev/)
[![Build Status](https://github.com/artemispados/NeuralLinearSolve.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/artemispados/NeuralLinearSolve.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/artemispados/NeuralLinearSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/artemispados/NeuralLinearSolve.jl)

`NeuralLinearSolve.jl` is a Julia package that uses a pretrained CNN to predict solve time ratios between solvers, returning the recommended solver.

Given a `SparseMatrixCSC`, the package generates a 64×64 SPY plot image and passes it through a frozen CNN to classify the matrix into one of three solver categories: **UMFPACK**, **KLU**, or **Pardiso**.

## Installation

```julia
using Pkg
Pkg.add("NeuralLinearSolve")
```

## Example Usage

```julia
using NeuralLinearSolve
using SparseArrays

A = sprand(1000, 1000, 0.01)

# Get the recommended solver
solver = predict_solver(A)
# e.g. :KLU

# Get relative speed scores for each solver (higher = more strongly recommended)
scores = predict_solver_scores(A)
# e.g. Dict(:UMFPACK => 0.0, :KLU => 1.2, :Pardiso => 0.3)
# The minimum score is always 0.0 (least recommended solver)
```

The CNN was trained on timing data from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/), comparing solve times for UMFPACK, KLU, and Pardiso across ~1000 sparse matrices. It achieved **94.2% test accuracy** on held-out matrices.

## Supported solvers

| Symbol | Solver | LinearSolve.jl algorithm |
|--------|--------|--------------------------|
| `:UMFPACK` | UMFPACK (default Julia sparse solver, via SuiteSparse) | `UMFPACKFactorization()` |
| `:KLU` | KLU (via SuiteSparse) | `KLUFactorization()` |
| `:Pardiso` | MKL Pardiso (Intel MKL) | `MKLPardisoFactorize()` |

## License

MIT
