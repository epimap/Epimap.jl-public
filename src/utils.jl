"""
    kron2d(A::AbstractMatrix, B::AbstractMatrix)

Implementation of `Base.kron` for matrices `A` and `B`.

This can be useful as a replacement of `Base.kron` because:
- GPU compatible (https://github.com/JuliaGPU/CUDA.jl/pull/814).
- Potentially faster.

## See also
- https://github.com/JuliaGPU/CUDA.jl/pull/814
"""
function kron2d(A::AbstractMatrix, B::AbstractMatrix)
    A4 = reshape(A, 1, size(A, 1), 1, size(A, 2))
    B4 = reshape(B, size(B, 1), 1, size(B, 2), 1)
    C4 = A4 .* B4
    return reshape(C4, size(A, 1) * size(B, 1), size(A, 2) * size(B, 2))
end


"""
    @map! f a b c ...

Replace variables with output of `f`.

Expands to `a = f(a); b = f(b); c = f(c); ...`.
"""
macro map!(f, args...)
    @gensym g

    exprs = []
    for a in args
        push!(exprs, :($a = $g($a)))
    end

    return esc(:($g = $f; $(exprs...)))
end

"""
    tonamedtuple(x::AbstractArray, ax::ComponentArrays.Axis)

Convert `x` to a `NamedTuple` accoridng to `ax`.

Useful in certain cases where you want to circumvent the use of `ComponentArray`, e.g.
using ForwardDiff.jl/ReverseDiff.jl/Tracker.jl.
"""
@generated function tonamedtuple(x::AbstractArray, ax::ComponentArrays.Axis)
    indexmap = ComponentArrays.indexmap(ax)
    names = keys(indexmap)

    vals = Expr(:tuple)
    for inds in indexmap
        push!(vals.args, :(Base.maybeview(x, $inds)))
    end

    return :(NamedTuple{$names}($vals))
end

issampling(context::DynamicPPL.AbstractContext) = issampling(DynamicPPL.NodeTrait(issampling, context), context)
issampling(::DynamicPPL.SamplingContext) = true
issampling(::DynamicPPL.IsLeaf, context) = false
issampling(::DynamicPPL.IsParent, context) = issampling(DynamicPPL.childcontext(context))
