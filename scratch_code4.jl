n_circs = 2
lin = false
using Flux
in_out_dim = 7
lin_dim = 1
function circ(x)
  length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
  x./sqrt(sum(x .* x))
end
println("double circular layer autoencoder being created")
encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
encodetobottleneck1 = Chain(Dense(in_out_dim, 2), circ)
encodetobottleneck2 = Chain(Dense(in_out_dim, 2), circ)
encodetobottleneck(x) = vcat(encodetobottleneck1(x), encodetobottleneck2(x))
decoder = Dense(n_circs*2 + lin, in_out_dim)
model = Chain(encodetobottleneck, decoder)
