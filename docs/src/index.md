# NeuralLinearSolve.jl: Neural Network-Based Linear Solver Selection

NeuralLinearSolve.jl uses a pretrained convolutional neural network (CNN) to predict
the fastest linear solver for a given sparse matrix. The sparse matrix is converted to
a 64x64 spy plot image, which is fed into the CNN to predict the optimal solver from
UMFPACK, KLU, and Pardiso.

## Installation

To install NeuralLinearSolve.jl, use the Julia package manager:

```julia
using Pkg
Pkg.add("NeuralLinearSolve")
```

## Quick Start

```julia
using NeuralLinearSolve
using SparseArrays

A = sprand(1000, 1000, 0.01)

# Get the predicted fastest solver
solver = predict_solver(A)

# Get the full probability distribution over solvers
probs = predict_solver_probs(A)
```

## API

```@docs
predict_solver
predict_solver_probs
```

## Contributing

  - Please refer to the
    [SciML ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://github.com/SciML/ColPrac)
    for guidance on PRs, issues, and other matters relating to contributing to SciML.

  - See the [SciML Style Guide](https://github.com/SciML/SciMLStyle) for common coding practices and other style decisions.
  - There are a few community forums:

      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Slack](https://julialang.org/slack/)
      + The #diffeq-bridged and #sciml-bridged channels in the
        [Julia Zulip](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
      + On the [Julia Discourse forums](https://discourse.julialang.org)
      + See also [SciML Community page](https://sciml.ai/community/)

## Reproducibility

```@raw html
<details><summary>The documentation of this SciML package was built using these direct dependencies,</summary>
```

```@example
using Pkg # hide
Pkg.status() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>and using this machine and Julia version.</summary>
```

```@example
using InteractiveUtils # hide
versioninfo() # hide
```

```@raw html
</details>
```

```@raw html
<details><summary>A more complete overview of all dependencies and their versions is also provided.</summary>
```

```@example
using Pkg # hide
Pkg.status(; mode = PKGMODE_MANIFEST) # hide
```

```@raw html
</details>
```

```@eval
using TOML
using Markdown
version = TOML.parse(read("../../Project.toml", String))["version"]
name = TOML.parse(read("../../Project.toml", String))["name"]
link_manifest = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
                "/assets/Manifest.toml"
link_project = "https://github.com/SciML/" * name * ".jl/tree/gh-pages/v" * version *
               "/assets/Project.toml"
Markdown.parse("""You can also download the
[manifest]($link_manifest)
file and the
[project]($link_project)
file.
""")
```
