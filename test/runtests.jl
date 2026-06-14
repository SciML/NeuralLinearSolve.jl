using Test
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @testset "NeuralLinearSolve.jl" begin

        @safetestset "predict_solver returns valid symbol" begin
            using NeuralLinearSolve
            using SparseArrays
            A = sprand(1000, 1000, 0.01)
            result = predict_solver(A)
            @test result isa Symbol
            @test result in (:UMFPACK, :KLU, :Pardiso)
        end

        @safetestset "predict_solver_probs returns valid probabilities" begin
            using NeuralLinearSolve
            using SparseArrays
            A = sprand(1000, 1000, 0.01)
            probs = predict_solver_probs(A)
            @test probs isa Dict{Symbol, Float32}
            @test isapprox(sum(values(probs)), 1.0, atol = 1.0e-5)
            @test all(v >= 0 for v in values(probs))
        end

        @safetestset "diagonal matrix prediction" begin
            using NeuralLinearSolve
            using SparseArrays
            using LinearAlgebra
            B = sparse(Diagonal(rand(500)))
            result = predict_solver(B)
            @test result isa Symbol
            @test result in (:UMFPACK, :KLU, :Pardiso)
        end

        @safetestset "spy plot generation" begin
            using NeuralLinearSolve
            using SparseArrays
            A = sprand(100, 100, 0.05)
            X = NeuralLinearSolve.matrix_to_spy(A)
            @test size(X) == (64, 64, 1, 1)
            @test eltype(X) == Float32
            @test all(0 .<= X .<= 1)
        end

    end
end

if GROUP == "QA"
    using Pkg
    Pkg.activate(joinpath(@__DIR__, "qa"))
    Pkg.instantiate()
    include(joinpath(@__DIR__, "qa", "qa.jl"))
end
