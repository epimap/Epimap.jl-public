using ArgParse, Dates, Serialization
using Epimap
using PyCall

# TODO: Add a bunch of arguments to allow further specification of the data,
# e.g. number of steps, the size of each timestep.

s = ArgParseSettings()
@add_arg_table s begin
    "out-path"
    help = "path where the generated model args should go"
    "--python"
    action = :store_true
    help = "if specified, will output model arg to pickled file with suffix '.pkl'"
    "--julia"
    action = :store_true
    help = "if specified, will output model arg to serialized file with suffix '.jls'"
end

parsed_args = parse_args(s)
out_path = parsed_args["out-path"]
for_python = parsed_args["python"]
for_julia = parsed_args["julia"]
for_both = (for_python && for_julia)

# Determinte the output paths.
directory_path = dirname(out_path)
filename = if isdirpath(out_path)
    println("Given output path is a directory; using default basename 'model_args'")
    "model_args"
else
    # If we're storing both, then we remove the extension.
    for_both ? splitext(basename(out_path))[1] : basename(out_path)
end

out_path = joinpath(directory_path, filename)

# If either an extension was not provided or we removed it
# because we're going to generate output for both files, we
# add the corresponding extension. If there already is an extension, we leave it.
out_path_python = isempty(splitext(out_path)[2]) ? out_path * ".pkl" : out_path
out_path_julia = isempty(splitext(out_path)[2]) ? out_path * ".jls" : out_path

# Load data and construct `setup_args`.
println("Loading data...")
data = Rmap.load_data()
println("OK!")

println("Constructing model args...")
setup_args = Rmap.setup_args(
    Rmap.rmap_naive, data, Float32,
    num_steps = 15,
    timestep = Week(1)
)
println("OK!")

if for_python
    # Convert the values in `setup_args` to corresponding python objects.
    print("Converting to Python objects...")
    setup_args_py = map(setup_args) do v
        if v isa AbstractArray
            # Reverse dims so get row-major ordering as is standard in Python.
            PyObject(PyReverseDims(Array(v)))
        else
            PyObject(v)
        end
    end;

    # Convert to a `Dict`, as it is more comon in Pyton.
    setup_args_py_dict = PyObject(Dict(pairs(setup_args_py)...));
    println("OK!")

    # Import `pickle`.
    pkl = pyimport("pickle")

    # Pickle the full `setup_args_py_dict`.
    print("Pickling model args to $(out_path_python)...")
    @pywith pybuiltin("open")(out_path_python, "wb+") as f begin
        pkl.dump(setup_args_py_dict, f)
    end
    println("OK!")
end

if for_julia
    print("Serializing model args to $(out_path_julia)...")
    serialize(out_path_julia, setup_args)
end
