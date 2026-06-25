using SciMLTesting
using NeuralLinearSolve
using JET
using Test

run_qa(
    NeuralLinearSolve;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    ei_kwargs = (;
        # `@load` is BSON's documented model-loading macro but is not declared
        # `public`/exported in BSON, so it trips all_explicit_imports_are_public.
        all_explicit_imports_are_public = (; ignore = (Symbol("@load"),)),
    ),
)
