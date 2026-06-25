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
    ),
    pages = [
        "Overview" => "index.md",
        "Ports: Rust vs Julia" => "ports.md",
        "Factorizations (faer)" => "factorizations.md",
    ],
    remotes = nothing,  # no GitHub remote configured locally → skip source-link generation
    warnonly = true,    # don't fail the build on cross-reference / docstring warnings
)
