using NeuralLinearSolve
using SparseArrays
using LinearAlgebra
using Test

@testset "NeuralLinearSolve.jl" begin

    @testset "predict_solver returns valid symbol" begin
        A = sprand(1000, 1000, 0.01)
        result = predict_solver(A)
        @test result isa Symbol
        @test result in (:UMFPACK, :KLU, :Pardiso)
    end

    @testset "predict_solver_probs returns valid probabilities" begin
        A = sprand(1000, 1000, 0.01)
        probs = predict_solver_probs(A)
        @test probs isa Dict{Symbol, Float32}
        @test isapprox(sum(values(probs)), 1.0, atol=1e-5)
        @test all(v >= 0 for v in values(probs))
    end

    @testset "diagonal matrix prediction" begin
        B = sparse(Diagonal(rand(500)))
        result = predict_solver(B)
        @test result isa Symbol
        @test result in (:UMFPACK, :KLU, :Pardiso)
    end

    @testset "spy plot generation" begin
        A = sprand(100, 100, 0.05)
        X = NeuralLinearSolve.matrix_to_spy(A)
        @test size(X) == (64, 64, 1, 1)
        @test eltype(X) == Float32
        @test all(0 .<= X .<= 1)
    end

end
