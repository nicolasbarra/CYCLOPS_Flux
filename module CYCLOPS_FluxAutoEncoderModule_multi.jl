module CYCLOPS_FluxAutoEncoderModule_multi

using Flux
using CSV

# circular activation function
function circ(z::Float64,zstar::Float64)
  z/(sqrt(z^2+zstar^2))
end

seed_homologues1 = CSV.read("Human_UbiquityCyclers.csv");

#encoder = Dense(outs1, 2, x -> x)
bottleneck = Dense(2, 2, circ)
#decoder = Dense(2, outs1, x -> x)

#model = Chain(encoder, bottleneck, decoder)

loss(x) = mse(model(x), x)


end
