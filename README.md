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
