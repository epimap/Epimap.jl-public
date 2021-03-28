using Adapt

"""
    FloatMaybeAdaptor{T<:Real}

Adaptor which will do its best to convert the storage-type into `T<:AbstractFloat`.

## Examples
```jldoctest
julia> using Epimap, Adapt

julia> x = (A = [1, 2, 3], B = [1.0, 2.0, 3.0], C = [1f0, 2f0, 3f0], a = 1, b = 1.0, c = 1f0);

julia> x_f32 = adapt(
           Epimap.FloatMaybeAdaptor{Float32}(),
           x
       )
(A = [1, 2, 3], B = Float32[1.0, 2.0, 3.0], C = Float32[1.0, 2.0, 3.0], a = 1, b = 1.0f0, c = 1.0f0)
```

"""
struct FloatMaybeAdaptor{T<:Real} end

Adapt.adapt_storage(::FloatMaybeAdaptor, x::Integer) = x
Adapt.adapt_storage(::FloatMaybeAdaptor{T}, x::Real) where {T} = T(x)
Adapt.adapt_storage(::FloatMaybeAdaptor{T}, x::AbstractArray{<:Integer}) where {T} = x
Adapt.adapt_storage(::FloatMaybeAdaptor{T}, x::AbstractArray{<:Real}) where {T} = T.(x)
