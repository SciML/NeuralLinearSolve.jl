using NeuralLinearSolve
using Aqua
using JET
using Test

@testset "Aqua" begin
    Aqua.test_all(NeuralLinearSolve)
end

@testset "JET" begin
    JET.test_package(NeuralLinearSolve; target_defined_modules = true)
end
