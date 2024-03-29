using Flux, CSV, Statistics, Distributed, DelimitedFiles, DataFrames
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

n_samples = length(alldata_times)
n_probes = length(alldata_probes)

timestamped_samples = setdiff(1:n_samples, findnotime(alldata_times))

cutrank = n_probes - MaxSeeds


CSV.write("error.csv", alldata_data, writeheader=false)
