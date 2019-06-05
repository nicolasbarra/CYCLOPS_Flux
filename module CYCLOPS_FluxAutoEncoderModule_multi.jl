module CYCLOPS_FluxAutoEncoderModule_multi

using Flux
using CSV
import Random

# circular activation function
function circ(z::Float64,zstar::Float64)
  z/(sqrt(z^2+zstar^2))
end

seed_homologues1 = CSV.read("Human_UbiquityCyclers.csv")
homologue_symbol_list1=seed_homologues1[2:end,2]

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

# seed Random Number Generator for reproducible results
Random.seed!(12345)

fullnonseed_data = read("Annotated_Unlogged_BA11Data.csv")

alldata_probes = fullnonseed_data[4:end, 1]
alldata_symbols = fullnonseed_data[4:end, 2]

alldata_times = fullnonseed_data[3, 4:end]
alldata_subjects = fullnonseed_data[2, 4:end]
alldata_samples = fullnonseed_data[1, 4:end]

alldata_data = fullnonseed_data[4:end, 4:end]


alldata_data = fullnonseed_data[4:end, 4:end]
alldata_data = Array{Float64}(alldata_data)

#encoder = Dense(outs1, 2, x -> x)
bottleneck = Dense(2, 2, circ)
#decoder = Dense(2, outs1, x -> x)

#model = Chain(encoder, bottleneck, decoder)

loss(x) = mse(model(x), x)


end
