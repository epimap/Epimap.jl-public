# We call `NNlib.pad_constant(u, Δsu, 0)` in our convolution methods to avoid
# the `NNlib.gen_pad` since it introduces type-instabilities. The two lines below are
# necessary to make the ChainRules-pullback for `NNlib.pad_constant` work with the
# direct `Δsu` we are using.
# TODO: Raise this isse over at NNlib and make PR with "fix"
# (a bit unclear if this is the below is the way to go though).
NNlib.gen_pad(pad::NTuple{N,Tuple{Int,Int}}, dims, _) where {N} = pad
NNlib.gen_pad(pad::NTuple{N,Tuple{Int,Int}}, dims::Colon, _) where {N} = pad
