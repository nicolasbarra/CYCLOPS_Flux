using Flux, CSV, Statistics, Distributed, Juno
import Random

@everywhere basedir = homedir()
@everywhere cd(basedir * "/github/CYCLOPS_Flux")
@everywhere include("CYCLOPS_CircularStatsModule.jl")
@everywhere include("CYCLOPS_PrePostProcessModule.jl")
@everywhere include("CYCLOPS_SeedModule.jl")
@everywhere include("CYCLOPS_SmoothModule_multi.jl")

# circular activation function
function circ(z::Float64,zstar::Float64)
    z/(sqrt(z^2+zstar^2))
end

# make all the columns (beginning at inputted column number) of a the alldata_data DataFrame of type Float64, not String since they are Numbers
function makefloat!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) == Array{String,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end

# find the samples that have no time stamp so you can remove them
function findnotime(df)
  r = []
  for row in 1:length(df)
    if typeof(df[row]) == Nothing
      append!(r, row)
    end
  end
  r
end

Frac_Var = 0.85  # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = 0.03  # Set Number of Dimensions of SVD so that incremetal fraction of variance of var is at least this much
N_best = 10  # Number of random initial conditions to try for each optimization
total_background_num = 10  # Number of background runs for global background refrence distribution (bootstrap). For real runs, this should be much higher.

seed_homologues1 = CSV.read("Human_UbiquityCyclers.csv")
homologue_symbol_list1 = seed_homologues1[1:end, 2]

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

# seed Random Number Generator for reproducible results
Random.seed!(12345)

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
alldata_probes = fullnonseed_data[3:end, 1]
alldata_symbols = fullnonseed_data[3:end, 2]
alldata_subjects = fullnonseed_data[1, 4:end]
alldata_times = fullnonseed_data[2, 4:end]
# first get the heade of the dataframe which has the samples as array of Strings
alldata_samples = String.(names(fullnonseed_data))
# then extract from that array only the headers that actually correspond with samples and not just the other headers that are there
alldata_samples = alldata_samples[4:end]

alldata_data = fullnonseed_data[3:end, 4:end]
makefloat!(1, alldata_data)
alldata_data = convert(Matrix, alldata_data)

n_samples = length(alldata_times)
n_probes = length(alldata_probes)

timestamped_samples = setdiff(1:n_samples, findnotime(alldata_times))

cutrank = n_probes - MaxSeeds

Seed_MinMean = (sort(vec(mean(alldata_data, dims = 2))))[cutrank]  # Note that this number is slightly different in new version versus old (42.88460199564358 here versus 42.88460199555892 in the old file). This is due to the fact that when the data is imported from the CSV it is automatically rounded after a certain number of decimal points.

#= This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets =#
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(fullnonseed_data, homologue_symbol_list1, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, Frac_Var, DFrac_Var, 30)
#= Data passed into Flux models must be in the form of an array of arrays where both the inner and outer arrays are one dimensional. This makes the array into an array of arrays. =#
norm_seed_data2 = mapslices(x -> [x], norm_seed_data1, dims=1)[:]

#= This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle =#
encoder = Dense(outs1, 2, x -> x)
bottleneck = Dense(2, 2, circ)
decoder = Dense(2, outs1, x -> x)

model = Chain(encoder, decoder)

loss(x) = Flux.mse(model(x), x)


Flux.@epochs 1 Flux.train!(loss, Flux.params(model), zip(norm_seed_data2), ADAM(), cb = Flux.throttle(@show(loss(norm_seed_data2)), 5))
