@testitem "specialfns" tags = [:specialfns] begin
    # SpecialFns is probe-first (see src/SpecialFns.jl): erf/gamma are expected to document-skip in
    # favour of SpecialFunctions.jl. This testitem is a placeholder until the probe
    # (bench/probe_specialfns.jl) decides whether to ship kernels here. When implemented, add
    # correctness vs SpecialFunctions.jl + per-submodule audit (:inlined,:noalloc,:trimsafe).
    using BlazingPorts: SpecialFns
    @test isempty(filter(n -> n != :SpecialFns && !startswith(string(n), "#"), names(SpecialFns; all = false)))
end
