# Epimap

Julia implementation of the Epimap models.

## Documentation
To generate the documention, navigate into the `docs/` directory and run:

```sh
julia --project -e 'using Pkg; pkg"dev .."; Pkg.instantiate(); include("make.jl");'
```

After the above finishes generating the documentation, it can be accessed by either:
1. Navigating to `docs/build/index.html` in your browser.
2. Run `julia --project -e 'using LiveServer; serve(dir="build")'` from the command-line.
3. Run `python3 -m http.server --bind localhost --directory build` from the command-line.

## Tests
To run the tests, you simply need to do
```julia
] test
```
in the Julia REPL. By default this will only run a subset of the available tests, specifically those that do not require external data and whatnot.

If you want to run the *full* test-suite, start the Julia REPL using:
```sh
EPIMAP_TEST_ALL=true julia --project
```
followed by
```julia
] test
```

When doing so, you'll also have to provide the location of the Rmap-data using an environment-variable, e.g.
```sh
export EPIMAP_TEST_ALL=true
export EPIMAP_RMAP_DATADIR=/path/to/rmap/data
julia --project
```
