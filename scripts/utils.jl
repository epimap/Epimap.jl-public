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
getcommit(run::String) = split(run, "-")[end]

"""
    getcommit(repo::LibGit2.GitRepo)

Return the commit ID of HEAD for `repo`.
"""
function getcommit(repo::LibGit2.GitRepo)
    githead = LibGit2.head(repo)
    return string(LibGit2.GitHash(LibGit2.peel(LibGit2.GitCommit, githead)))
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