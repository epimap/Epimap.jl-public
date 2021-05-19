using Epimap, Dates
using PyCall

const out_path = "setup_args.pkl"

# Load data and construct `setup_args`.
print("Loading data...")
data = Rmap.load_data()
println("OK!")

print("Constructing model args...")
setup_args = Rmap.setup_args(
    Rmap.rmap_naive, data, Float32,
    num_steps = 15,
    timestep = Week(1)
)
println("OK!")

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
print("Pickling model args to $(out_path)...")
@pywith pybuiltin("open")(out_path, "wb+") as f begin
    pkl.dump(setup_args_py_dict, f)
end
println("Done!")
