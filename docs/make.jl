using Documenter
using Literate
using CairoMakie  # so @example blocks can render figures during the docs build
using SnowingOcean

# Convert the Literate example sources in examples/ into markdown pages
example_dir = joinpath(@__DIR__, "..", "examples")
literated_dir = joinpath(@__DIR__, "src", "literated")

examples = [
    "melting_under_ice.jl",
]

for example in examples
    Literate.markdown(joinpath(example_dir, example), literated_dir; documenter=true)
end

example_pages = ["literated/" * first(splitext(e)) * ".md" for e in examples]

makedocs(
    sitename = "SnowingOcean.jl",
    modules = [SnowingOcean],
    pages = [
        "Home" => "index.md",
        "Examples" => example_pages,
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
