@testitem "specialfns" tags = [:specialfns] begin
    # SpecialFns is probe-first (see src/SpecialFns.jl): erf/gamma are expected to document-skip in
    # favour of SpecialFunctions.jl. This testitem is a placeholder until the probe
    # (bench/probe_specialfns.jl) decides whether to ship kernels here. When implemented, add
    # correctness vs SpecialFunctions.jl + per-submodule audit (:inlined,:noalloc,:trimsafe).
    using BlazingPorts: SpecialFns
    @test isempty(filter(n -> n != :SpecialFns && !startswith(string(n), "#"), names(SpecialFns; all = false)))
end

@testitem "specialfns_strictmode" tags = [:specialfns] begin
    # Probe-first: empty shell until the probe ships erf/gamma kernels. The audit is wired NOW so
    # any future kernel lands under (:inlined, :noalloc, :trimsafe) automatically instead of
    # shipping unchecked (an empty module audits to zero checks — harmless).
    using BlazingPorts: SpecialFns
    using StrictMode, AllocCheck, JET
    fs = audit(SpecialFns; sweep = true, guarantees = (:inlined, :noalloc, :trimsafe), io = devnull)
    @test nfailures(fs) == 0
end
