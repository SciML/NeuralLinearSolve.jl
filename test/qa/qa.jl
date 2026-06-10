using NeuralLinearSolve
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(NeuralLinearSolve; deps_compat = (; check_extras = false))
    # deps_compat extras check fails: root Project.toml lists `Pkg` under [extras]
    # without a [compat] entry. Tracked at:
    # https://github.com/SciML/NeuralLinearSolve.jl/issues/10
    @test_broken false  # Aqua deps_compat extras: missing [compat] for `Pkg` — see https://github.com/SciML/NeuralLinearSolve.jl/issues/10
end

@testset "JET" begin
    JET.test_package(NeuralLinearSolve; target_defined_modules = true)
end
