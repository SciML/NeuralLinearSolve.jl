# NeuralLinearSolve.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/NeuralLinearSolve/stable/)

[![codecov](https://codecov.io/gh/SciML/NeuralLinearSolve.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/SciML/NeuralLinearSolve.jl)
[![Build Status](https://github.com/SciML/NeuralLinearSolve.jl/workflows/CI/badge.svg)](https://github.com/SciML/NeuralLinearSolve.jl/actions?query=workflow%3ACI)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

`NeuralLinearSolve.jl` is a Julia package that uses a pretrained CNN to predict the fastest direct linear solver for a given sparse matrix.

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

# Get the full probability distribution over solvers
probs = predict_solver_probs(A)
# e.g. Dict(:UMFPACK => 0.05, :KLU => 0.91, :Pardiso => 0.04)
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
