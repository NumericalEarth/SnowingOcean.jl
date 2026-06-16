using Documenter
using SnowingOcean

makedocs(
    sitename = "SnowingOcean.jl",
    modules = [SnowingOcean],
    pages = [
        "Home" => "index.md",
        "Library" => "library.md",
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://NumericalEarth.github.io/SnowingOcean.jl/stable/",
    ),
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/NumericalEarth/SnowingOcean.jl",
    devbranch = "main",
    push_preview = true,
)
