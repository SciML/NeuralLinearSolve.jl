using SafeTestsets

@safetestset "Aqua" begin
    using NeuralLinearSolve
    using Aqua
    using Test
    Aqua.test_all(NeuralLinearSolve; deps_compat = (; check_extras = false))
    # deps_compat extras check fails: root Project.toml lists `Pkg` under [extras]
    # without a [compat] entry. Tracked at:
    # https://github.com/SciML/NeuralLinearSolve.jl/issues/10
    @test_broken false  # Aqua deps_compat extras: missing [compat] for `Pkg` — see https://github.com/SciML/NeuralLinearSolve.jl/issues/10
end

@safetestset "JET" begin
    using NeuralLinearSolve
    using JET
    using Test
    JET.test_package(NeuralLinearSolve; target_defined_modules = true)
end
