import DSP, AbstractFFTs


"""
    conv(x, w)

Convolve `x` with filter/window `w`. Assumes `w` is "smallest" size.

## Notes
- We return the left-padded convolution, NOT the symmetrically padded convolution.
  That is, we compute `x[t - i] * w[i] for  i = 1:t` only for `t = 1, ..., length(x)`
"""
conv(x, w) = conv_nnlib(x, w)

# Convenience method for the Matrix × Vector cases.
for f in [:conv_fft, :conv_dsp]
    @eval $f(x::AbstractMatrix, w::AbstractVector) = $f(x, reshape(w, 1, :))
end

"""
    conv_nnlib(x::AbstractVector, w::AbstractVector)
    conv_nnlib(x::AbstractMatrix, w::AbstractVector)

Convolves `x` with filter/window `w` using `NNlib.conv`.

## Notes
- Important!!! `NNlib.conv` wants to work with shapes of the form `(width, height, channels, batch_size)`, or in terms more related to our problem, `(num_times, num_regions, 1, 1)`. At the moment [2021-03-26 Fri] we generally work with sizes of the form `(num_regions, num_times)`. As a result, this method will first transpose `x` and `w` (if they are matrices, of course) before calling `NNlib.conv`, and so we could potentially speed this up a tiny bit by working with the transposed format.
"""
function conv_nnlib(x::AbstractVector, w::AbstractVector)
    x_arr = reshape(x, :, 1, 1)
    w_arr = reshape(w, :, 1, 1)

    # HACK: Honestly don't understand why in the case of 1D convolution we only
    # want padding to be either 1d or 2d, not 3d. But it complains if we don't
    # and it get's the correct result.
    return NNlib.conv(x_arr, w_arr; pad = (size(w_arr, 1) - 1, 0))[:, 1, 1]
end

# NOTE: we treat the input as 
function conv_nnlib(x::AbstractMatrix, w::AbstractVector)
    # 2D convolution should be 4D, so we add two additional
    # dimensions (channels and batch-size) to the end of the inputs.

    # Materialize the transposes because otherwise Zygote.jl will complain.
    x_arr = reshape(Array(transpose(x)), size(x, 2), 1, size(x, 1))
    w_arr = reshape(w, length(w), 1, 1)

    # Result, dropping the last two dimensions that we added.
    res = NNlib.conv(x_arr, w_arr, pad = (size(w_arr, 1) - 1, 0))

    # Recover original shape.
    return transpose(res[:, 1, :])
end

"""
    conv_dsp(x, w)

Convolves `x` with filter/window `w` using `DSP.conv`.

## Notes
- This is NOT an AD-friendly implementation!!!
"""

conv_dsp(x::AbstractArray, w::AbstractArray) = DSP.conv(x, w)[CartesianIndices(x)]


import DSP: AbstractFFTs
function fftpad(u::AbstractArray, su, outsize, dims)
    # HACK: This seems way to hacky. Surely there exists a better solution?
    j = 0
    Δsu = ntuple(length(su)) do i
        i in dims ? (0, outsize[j += 1] - su[i]) : (0, 0)
    end

    # Using `NNlib.pad_constant` rather than `NNlib.pad_zeros` since this will
    # circumvent the call to `NNlib.gen_pad` which seemst to introduce
    # type-instabilities.
    return NNlib.pad_constant(u, Δsu, 0)
end

function fftpad(u::AbstractArray, su, outsize)
    Δsu = ntuple(length(su)) do i
        (0, outsize[i] - su[i])
    end

    # Using `NNlib.pad_constant` rather than `NNlib.pad_zeros` since this will
    # circumvent the call to `NNlib.gen_pad` which seemst to introduce
    # type-instabilities.
    return NNlib.pad_constant(u, Δsu, 0)
end

"""
    conv_fft(x, w)

Convolves `x` with filter/window `w` by padded multiplication in Fourier space.

## Notes
- The FFT transform is provided by AbstractFFTs.jl, which, since we are using DSP.jl,
uses FFTW.jl under the hood.
"""
function conv_fft(u::AbstractArray, v::AbstractArray, dims)
    su, sv = size(u), size(v)
    outsize = su[dims] .+ sv[dims] .- 1

    # Actual stuff
    upad = fftpad(u, su, outsize, dims)
    vpad = fftpad(v, sv, outsize, dims)

    # # Set up the `fft` plan
    # P = FFTW.plan_fft(upad, dims)

    # `real` here ensures that the pullback through `fft` will be projected to real

    # uft = P * real(upad)
    # vft = P * real(vpad)

    uft = AbstractFFTs.fft(real(upad), dims)
    vft = AbstractFFTs.fft(real(vpad), dims)

    return real(AbstractFFTs.ifft(uft .* vft, dims))[CartesianIndices(u)]
end

function conv_fft(u::AbstractArray, v::AbstractArray)
    su, sv = size(u), size(v)
    outsize = su .+ sv .- 1

    # Actual stuff
    upad = fftpad(u, su, outsize)
    vpad = fftpad(v, sv, outsize)

    # # Set up the `fft` plan
    # P = FFTW.plan_fft(upad)

    # `real` here ensures that the pullback through `fft` will be projected to real

    # uft = P * real(upad)
    # vft = P * real(vpad)

    uft = AbstractFFTs.fft(real(upad))
    vft = AbstractFFTs.fft(real(vpad))

    return real(AbstractFFTs.ifft(uft .* vft))[CartesianIndices(u)]
end
