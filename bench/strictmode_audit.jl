# Per-submodule StrictMode audit (mirrors PureFFT's bench/strictmode_audit.jl, but one audit PER
# crate submodule rather than a whole-package sweep — so each crate's guarantees are reported and
# gated independently).
#
# Run (checks must be enabled — a compile-time Preference, already set in bench/Project.toml):
#   julia --project=bench bench/strictmode_audit.jl

using BlazingPorts, StrictMode
using AllocCheck, JET   # StrictMode's analysis backend is a weak-dep extension — load it for the sweep

StrictMode.checks_enabled() || error("StrictMode checks disabled — set [preferences.StrictMode] checks_enabled=true")
StrictMode.backend_available() || error("StrictMode analysis backend not loaded — need `using AllocCheck, JET`")

# Warm each submodule's hot surface so the usage-driven sweep sees real compiled methods.
function warm()
    SM = BlazingPorts.SmallMatrix
    a = SM.Vec3(1.0, 2.0, 3.0); b = SM.Vec3(4.0, 5.0, 6.0)
    SM.dot(a, b); SM.cross(a, b); SM.norm(a); SM.normalize(a); a + b; 2.0 * a
    v = SM.Vec4(1.0, 2.0, 3.0, 4.0)
    m = SM.Mat4(v, v, v, v); m * v; m * m
    return nothing
end
warm()

# (submodule, guarantees) per the triage table. Empty submodules (probe-first, not yet implemented)
# audit to zero checks — harmless.
const TARGETS = [
    (BlazingPorts.SmallMatrix, (:typestable, :noalloc)),
    (BlazingPorts.SpecialFns, (:inlined, :noalloc, :trimsafe)),
    (BlazingPorts.MatrixMultiply, (:vectorized, :noalloc)),
]

total_fail = 0
for (M, guarantees) in TARGETS
    println("\n══ audit $M  guarantees=$(guarantees) ", "═"^20)
    fs = audit(M; sweep = true, guarantees = guarantees, format = :text)
    nf = nfailures(fs)
    global total_fail += nf
    println("  → $(length(fs)) (method, guarantee) checks, $nf failure(s).")
end

println()
total_fail == 0 || error("StrictMode found $total_fail failure(s) across submodules.")
println("All implemented submodules satisfy their declared StrictMode guarantees. ✓")
