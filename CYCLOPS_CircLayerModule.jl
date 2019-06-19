module CYCLOPS_CircLayerModule

#= The following code is a modification of the Flux Dense layer defintion to create a new kind of layer named a Circ layer. This was created to be used as the circular layer in the autoencoder, but, ultimately, a different strategy was chosen to implement the circular nodes. However, this file has been retained in case this method is returned to later. =#

using Flux

function findpartner(x::Integer) # find index of partnered node
    grp = div(x - 1, 2)
    elm = mod(x - 1, 2)
    pelm = (1 - elm)
    partner = 1 + (grp) * 2 + pelm

    partner
end

glorot_uniform(dims...) = (rand(Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))

struct Circ{F,S,T,P,N}
  W::S
  b::T
  p::P
  n::N
  σ::F
end

Circ(W, b, p) = Circ(W, b, p, n, fun)

function Circ(in::Integer, out::Integer, circs::Integer, σ = fun;
               initW = glorot_uniform, initb = zeros, initp = zeros, initn = 0)
  return Circ(param(initW(out, in)), param(initb(out)), param(initp(in)), param(initn(circs)), σ)
end

Flux.@treelike Circ

function (a::Circ)(x::AbstractArray)
  W, b, p, n, σ = a.W, a.b, a.p, a.n, a.σ
  circ.(W*x .+ b, for i in 1:n
            p[i]=findpartner(i)
    end)
end


function Base.show(io::IO, l::Circ)
  print(io, "Circ(", size(l.W, 2), ", ", size(l.W, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


# Try to avoid hitting generic matmul in some simple cases
# Base's matmul is so slow that it's worth the extra conversion to hit BLAS
(a::Circ{<:Any,W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  invoke(a, Tuple{AbstractArray}, x)

(a::Circ{<:Any,W})(x::AbstractArray{<:Real}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
  a(T.(x))

end  # module Circ_Layer
