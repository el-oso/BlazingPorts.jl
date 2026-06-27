# Pure-markdown DocumenterVitepress site (no `using BlazingPorts` / `@docs` blocks, so it builds
# independently of the package source). Local render: `julia --project=docs docs/make.jl`.
using Documenter, DocumenterVitepress

makedocs(;
    sitename = "BlazingPorts.jl",
    authors = "el-oso",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/el-oso/BlazingPorts.jl",
        devbranch = "master",
        devurl = "dev",
        sidebar_drawer = true,   # render the navbar SidebarDrawerToggle (collapse the left menu on desktop)
    ),
    pages = [
        "Overview" => "index.md",
        "Ports: Rust vs Julia" => "ports.md",
        "Factorizations (faer)" => "factorizations.md",
    ],
    remotes = nothing,  # no GitHub remote configured locally → skip source-link generation
    warnonly = true,    # don't fail the build on cross-reference / docstring warnings
)

# Deploy the built Vitepress site to gh-pages. A no-op locally (deploydocs only acts in CI); in the
# Documenter GitHub Action it pushes via GITHUB_TOKEN. Must be DocumenterVitepress.deploydocs (not
# Documenter.deploydocs) or the Vitepress site 404s.
DocumenterVitepress.deploydocs(;
    repo = "github.com/el-oso/BlazingPorts.jl",
    target = joinpath(@__DIR__, "build"),
    devbranch = "master",
    branch = "gh-pages",
    push_preview = true,
)
