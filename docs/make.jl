using Epimap
using Documenter

DocMeta.setdocmeta!(Epimap, :DocTestSetup, :(using Epimap); recursive=true)

makedocs(;
    modules=[Epimap],
    authors="Tor Erlend Fjelde, Micheal J. Hutchinson, Yee Whye Teh, Hong Ge",
    repo="https://github.com/epimap/Epimap.jl/blob/{commit}{path}#{line}",
    sitename="Epimap.jl",
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://epimap.github.io/Epimap.jl",
        assets=String[],
        mathengine=Documenter.Writers.HTMLWriter.MathJax3()
    ),
    pages=[
        "Home" => "index.md",
    ],
)
