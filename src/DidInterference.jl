module DidInterference

using DataFrames
using GLM
using LinearAlgebra
using Statistics
using StatsBase
using Distributions: Normal, quantile

export did_int_2x2, did_int_dynamic, did_int_staggered

include("dr_atte.jl")
include("dr_atte_mult.jl")
include("did_int_2x2.jl")
include("did_int_dynamic.jl")
include("did_int_staggered.jl")

end # module
