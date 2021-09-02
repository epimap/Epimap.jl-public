using ArgParse, DrWatson

using Pkg: Pkg
using LibGit2: LibGit2
"""
    pkgversion(m::Module)

Return version of module `m` as listed in its Project.toml.
"""
function pkgversion(m::Module)
    projecttoml_path = joinpath(dirname(pathof(m)), "..", "Project.toml")
    return Pkg.TOML.parsefile(projecttoml_path)["version"]
end

"""
    @trynumerical f(x)
    @trynumerical max_tries f(x)

Attempts to evaluate `f(x)` until either
1. `f(x)` successfully evaluates,
2. no `InexactError` is thrown, or
3. we have attempted evaluation more than `max_tries` times.

Errors other than `InexactError` will be thrown as usual.
"""
macro trynumerical(expr)
    return esc(trynumerical(10, expr))
end
macro trynumerical(max_tries, expr)
    return esc(trynumerical(max_tries, expr))
end
function trynumerical(max_tries, expr)
    @gensym i result
    return quote
        local $result
        for $i = 1:$max_tries
            try
                $result = $expr
                break
            catch e
                if e isa $(InexactError)
                    # Yes this is a bit weird. It avoids clashes with
                    # namespace, e.g. there could be a `i` defined in `expr`
                    # but still allows us to do string interpolation.
                    let i = $i
                        @info "Failed on attempt $i due to numerical error"
                    end
                    continue
                else
                    rethrow(e)
                end
            end
        end
        $result
    end
end

###########################
# Reproducibility related #
###########################
"""
    default_name(; include_commit_id=false)

Construct a name from either repo information or package version
of `DynamicPPL`.

If the path of `DynamicPPL` is a git-repo, return name of current branch,
joined with the commit id if `include_commit_id` is `true`.

If path of `DynamicPPL` is _not_ a git-repo, it is assumed to be a release,
resulting in a name of the form `release-VERSION`.
"""
function default_name(mod; include_commit_id=true)
    mod_path = abspath(joinpath(dirname(pathof(mod)), ".."))

    # Extract branch name and commit id
    local name
    try
        githead = LibGit2.head(LibGit2.GitRepo(mod_path))
        branchname = LibGit2.shortname(githead)

        name = replace(branchname, "/" => "_")
        if include_commit_id
            gitcommit = LibGit2.peel(LibGit2.GitCommit, githead)
            commitid = string(LibGit2.GitHash(gitcommit))
            name *= "-$(commitid)"
        end
    catch e
        if e isa LibGit2.GitError
            @info "No git repo found for $(mod_path); extracting name from package version."
            name = "release-$(pkgversion(mod))"
        else
            rethrow(e)
        end
    end

    return name
end

"""
    getcommit(run)

Return the commit ID from the run name.

Assumes `name` came form [`default_path`](@ref).
"""
getcommit(run::String) = LibGit2.GitHash(split(run, "-")[end])

"""
    getcommit(repo::LibGit2.GitRepo)

Return the commit ID of HEAD for `repo`.
"""
function getcommit(repo::LibGit2.GitRepo)
    githead = LibGit2.head(repo)
    return LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, githead))
end

"""
    available_runs(; prefix=nothing, commit=nothing, full_path=false)

Return available runs.

# Keyword arguments
- `prefix`: filters based on the prefix of the runs.
- `commit`: filters based on the commit id from which the run is produced.
- `full_path`: if `true`, the full path to each run will be returned rather
  than only their directory names.
"""
function available_runs(; prefix=nothing, commit=nothing, full_path=false)
    runs = readdir(projectdir("intermediate"))
    if prefix !== nothing
        filter!(Base.Fix2(startswith, prefix), runs)
    end

    if commit !== nothing
        commit_ids = map(runs) do run
            last(split(run, "-"))
        end
        indices = findall(Base.Fix2(startswith, commit), commit_ids)
        runs = runs[indices]
    end

    if full_path
        map!(runs, runs) do run
            projectdir("intermediate", run)
        end
    end

    return runs
end

"""
    interactive_checkout_maybe(source, repodir=projectdir())
    interactive_checkout_maybe(source_commit::LibGit2.GitHash, repodir=projectdir())

Check if commit of `source` matches current HEAD of `repodir`.

If `source` is specified instead of `source_commit`, then `getcommit(source)` is used.
"""
function interactive_checkout_maybe(source, repodir=projectdir())
    return interactive_checkout_maybe(getcommit(source), repodir)
end
function interactive_checkout_maybe(
    source_commit::LibGit2.GitHash,
    repodir=projectdir()
)
    repo = LibGit2.GitRepo(repodir)

    if source_commit != getcommit(repo)
        print(
            "Run came from $(source_commit) but HEAD is ",
            "currently pointing to $(getcommit(repo)); ",
            "do you want to checkout the correct branch? [y/N]: "
        )
        answer = readline()
        if lowercase(answer) == "y"
            if LibGit2.isdirty(repo)
                error("HEAD is dirty! Please stash or commit the changes.")
            end
            LibGit2.checkout!(repo, string(source_commit))
        else
            error("Add flag --ignore-commit to avoid this prompt/check.")
        end
    elseif LibGit2.isdirty(repo)
        print("HEAD is dirty! Are you certain you want to continue? [y/N]: ")
        answer = readline()
        if lowercase(answer) != "y"
            exit(1)
        end
    end

    return nothing
end

#######################
# ArgParse.jl related #
#######################
"""
    add_default_args!(s::ArgParseSettings)

Add generally applicable arguments to `s`.

In particular, it adds the following arguments:
- `--ignore-commit`
- `--verbose`
"""
function add_default_args!(s::ArgParseSettings)
    @add_arg_table! s begin
        "--ignore-commit"
        help = "If specified, no check to ensure that we're working with the correct version of the package is performed."
        action = :store_true
        "--verbose"
        help = "If specified, additional info will be printed."
        action = :store_true
    end
    return s
end

"""
    add_default_sampling_args!(s::ArgParseSettings)

Add sampling related arguments to `s`.

In particular, it adds the following arguments:
- `--nsamples`
- `--nadapts`
- `--thin`
"""
function add_default_sampling_args!(s::ArgParseSettings)
    @add_arg_table! s begin
        "--nsamples"
        help = "Number of samples to produce."
        default = 1000
        arg_type = Int
        "--nadapts"
        help = "Number of adaptation steps to take, if applicable."
        default = 1000
        arg_type = Int
        "--thin"
        help = "Thinning interval to use."
        default = 1
        arg_type = Int
    end
    return s
end

"""
    add_sampling_postprocessing_args!(s::ArgParseSettings)

Add arguments related to postprocessing of runs to `s`.

In particular, it adds the following arguments:
"""
function add_sampling_sampling_args!(s::ArgParseSettings)
    return s
end

"""
    @parse_args s
    @parse_args s _args

Calls `parse_args` on `s` and `_args`, with `_args = ARGS` if `_args` is not defined.

This is convenient when wanting to run a script using `include` instead of from the
command line, for example allowing temporary halting of a long-running process to 
save some results, and then resume.
"""
macro parse_args(argparsesettings, argsvar=:_args)
    return quote
        if !($(Expr(:escape, Expr(:isdefined, argsvar))))
            $(esc(argsvar)) = $(esc(:ARGS))
        end

        $(ArgParse.parse_args)($(esc(argsvar)), $argparsesettings)
    end
end
