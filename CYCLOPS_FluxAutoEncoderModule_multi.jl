module CYCLOPS_FluxAutoEncoderModule_multi

using Flux, CSV, Statistics
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

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")

alldata_probes = fullnonseed_data[3:end, 1]
alldata_symbols = fullnonseed_data[4:end, 2]

alldata_times = fullnonseed_data[3, 4:end]
alldata_subjects = fullnonseed_data[2, 4:end]
alldata_samples = fullnonseed_data[1, 4:end]

alldata_data = fullnonseed_data[4:end, 4:end]

# make all the columns of type Float64, not String since they are Numbers
function makefloat!(df)
    for col in 1:size(alldata_data)[2]
        if typeof(alldata_data[:, col]) == Array{String,1}
            alldata_data[:, col] = map(x -> tryparse(Float64, x), alldata_data[:, col])
        end
    end
end

makefloat!(alldata_data)

alldata_data = convert(Matrix, alldata_data)

n_samples = length(alldata_times)
n_probes = length(alldata_probes)

# find the samples that have no time stamp so you can remove them
function findnotime(df)
  r = []
  for row in 1:length(df)
    if typeof(df[row]) == String
      append!(r, row)
    end
  end
  r
end

timestamped_samples = setdiff(1:n_samples, findnotime(alldata_times))

cutrank = n_probes-MaxSeeds

Seed_MinMean = (sort(vec(mean(alldata_data, dims=2))))[cutrank]
println(Seed_MinMean) # Note that this number is slightly different in new version versus old (42.889112582772285 here versus 42.88460199555892 in the old file) this is likely due to the fact that my method removes null values better (maybe?))

#= This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets =#

#=
#encoder = Dense(outs1, 2, x -> x)
bottleneck = Dense(2, 2, circ)
#decoder = Dense(2, outs1, x -> x)

#model = Chain(encoder, bottleneck, decoder)

loss(x) = mse(model(x), x)

=#
end
