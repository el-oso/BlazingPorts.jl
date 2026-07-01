# Structural backstops for CLAUDE.md rules 4/6 — these make "StrictMode is applied per crate"
# a red test instead of a convention an agent can silently drift from.

@testitem "strictmode enforced (checks on, or red in CI)" tags = [:strictmode_meta] begin
    # With checks disabled every @assert_* in this suite expands to the bare call and the crate
    # testitems pass VACUOUSLY. assert_enabled() errors under CI in that state; locally it just
    # reports. One item guards the whole repo.
    using StrictMode
    @test StrictMode.assert_enabled()
end

@testitem "strictmode meta: every crate test file audits its crate" tags = [:strictmode_meta] begin
    # CLAUDE.md rule 6: one StrictMode audit per submodule. This meta-test fails for any crate
    # whose test file carries no actual audit/assert — a TODO comment doesn't count (comments are
    # stripped before matching; two crates shipped exactly that way and stayed green for weeks).
    testdir = @__DIR__
    srcdir = normpath(joinpath(testdir, "..", "src"))
    crates = [splitext(f)[1] for f in readdir(srcdir) if endswith(f, ".jl") && f != "BlazingPorts.jl"]
    @test !isempty(crates)
    for crate in crates
        tf = joinpath(testdir, lowercase(crate) * "_tests.jl")
        @test isfile(tf)
        isfile(tf) || continue
        code = join(filter(l -> !startswith(lstrip(l), "#"), readlines(tf)), "\n")
        has_audit = occursin("audit(", code) || occursin(r"@assert_\w+", code)
        has_audit || @info "crate `$crate`: no StrictMode audit/@assert_* in $(basename(tf)) — " *
            "add a *_strictmode testitem (CLAUDE.md rule 6)"
        @test has_audit
    end
end
