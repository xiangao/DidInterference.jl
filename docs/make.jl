using Pkg
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using DidInterference

DocMeta.setdocmeta!(DidInterference, :DocTestSetup,
                    :(using DidInterference); recursive = true)

makedocs(
    sitename = "DidInterference.jl",
    modules  = [DidInterference],
    format   = Documenter.HTML(
        edit_link = nothing,
        repolink  = "https://github.com/xiangao/DidInterference.jl",
    ),
    pages = [
        "Home"        => "index.md",
        "Vignettes"   => [
            "Getting Started"      => "vignettes/01_getting_started.md",
            "Staggered Adoption"   => "vignettes/02_staggered.md",
            "Count Outcomes"       => "vignettes/03_multiplicative.md",
            "Decomposition"        => "vignettes/04_decomposition.md",
        ],
        "Reference"   => "reference.md",
    ],
    warnonly = true,
    checkdocs = :none,
    remotes  = nothing,
)

deploydocs(
    repo         = "github.com/xiangao/DidInterference.jl.git",
    devbranch    = "master",
    push_preview = false,
)
