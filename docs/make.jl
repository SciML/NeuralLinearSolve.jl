using NeuralLinearSolve
using Documenter

mkpath("./docs/src/assets")
cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

DocMeta.setdocmeta!(NeuralLinearSolve, :DocTestSetup, :(using NeuralLinearSolve); recursive = true)

include("pages.jl")

makedocs(
    sitename = "NeuralLinearSolve.jl",
    authors = "Chris Rackauckas",
    modules = [NeuralLinearSolve],
    clean = true, doctest = false, linkcheck = true,
    warnonly = [:docs_block, :missing_docs],
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/NeuralLinearSolve/stable/"
    ),
    pages = pages
)

deploydocs(;
    repo = "github.com/SciML/NeuralLinearSolve.jl",
    push_preview = true
)
