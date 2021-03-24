using Epimap
using Documenter

DocMeta.setdocmeta!(Epimap, :DocTestSetup, :(using Epimap); recursive=true)

makedocs(;
    modules=[Epimap],
    authors="Tor Erlend Fjelde, Micheal J. Hutchinson, Yee Whye Teh, Hong Ge",
    repo="https://github.com/torfjelde/Epimap.jl/blob/{commit}{path}#{line}",
    sitename="Epimap.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://torfjelde.github.io/Epimap.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/torfjelde/Epimap.jl",
)
