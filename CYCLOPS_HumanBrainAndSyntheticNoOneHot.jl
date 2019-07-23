using Flux, CSV, Statistics, Distributed, Juno, PyPlot, BSON, Revise, DataFrames

import Random

@everywhere basedir = homedir()
@everywhere cd(basedir * "/github/CYCLOPS_Flux")
@everywhere include("CYCLOPS_CircularStatsModule.jl")
@everywhere include("CYCLOPS_PrePostProcessModule.jl")
@everywhere include("CYCLOPS_SeedModule.jl")
@everywhere include("CYCLOPS_SmoothModule_multi.jl")
@everywhere include("CYCLOPS_TrainingModule.jl")
@everywhere include("CYCLOPS_FluxAutoEncoderModule.jl")
@everywhere cd(basedir * "/Downloads/Research")

# Find the samples that have no integer time stamp (so their type will be string) so you can remove them.
function findNAtime(df)
  r = []
  for row in 1:length(df)
    if typeof(df[row]) == String
      append!(r, row)
    end
  end

  r
end

Frac_Var = 0.85  # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = 0.03  #= Set Number of Dimensions of SVD so that incremetal fraction of
variance of var is at least this much=#
N_best = 10  # Number of random initial conditions to try for each optimization
total_background_num = 10  #= Number of background runs for global background refrence
distribution (bootstrap). For real runs, this should be much higher. =#

homologue_symbol_list = CSV.read("Human_UbiquityCyclers.csv")[1:end, 2]

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

# seed Random Number Generator for reproducible results
Random.seed!(12345)

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
fullnonseed_data_syn =
CSV.read("Annotated_Unlogged_BA11Data_r10_v1.csv")
deletecols!(fullnonseed_data_syn, [2, 3])
fullnonseed_data_joined = join(fullnonseed_data, fullnonseed_data_syn, on = :Column1, makeunique = true)
alldata_probes = fullnonseed_data_joined[3:end, 1]
alldata_symbols = fullnonseed_data_joined[3:end, 2]
alldata_subjects = fullnonseed_data_joined[1, 4:end]
alldata_times = fullnonseed_data_joined[2, 4:end]
# first get the head of the dataframe which has the samples as array of Strings and then extract from that array only the headers that actually correspond with samples and not just the other headers that are there
alldata_samples = String.(names(fullnonseed_data_joined))[4:end]

alldata_data = fullnonseed_data_joined[3:end, 4:end]
CYCLOPS_PrePostProcessModule.makefloat!(alldata_data)
alldata_data = convert(Matrix, alldata_data)

n_samples = length(alldata_times)

timestamped_samples = setdiff(1:n_samples, findNAtime(alldata_times))

alldata_times = (Vector(alldata_times))

truetimes = mod.(Array{Float64}(alldata_times[timestamped_samples]), 24)

n_probes = length(alldata_probes)
cutrank = n_probes - MaxSeeds

Seed_MinMean = (sort(vec(mean(alldata_data, dims = 2))))[cutrank]  #= Note that this number
is slightly different in new version versus old (42.88460199564358 here versus
42.88460199555892 in the old file). This is due to the fact that when the data is imported
from the CSV it is automatically rounded after a certain number of decimal points. =#

#= This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets =#
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(alldata_data, homologue_symbol_list, alldata_symbols, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, Frac_Var, DFrac_Var, 30)

#= Data passed into Flux models must be in the form of an array of arrays where both the
inner and outer arrays are one dimensional. This makes the array into an array of arrays. =#
norm_seed_data2 = mapslices(x -> [x], norm_seed_data1, dims=1)[:]

#= This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle =#
n_circs = 1  # set the number of circular layers in bottleneck layer
lin = false  # set the number of linear layers in bottleneck layer
lin_dim = 1  # set the in&out dimensions of the linear layers in bottleneck layer

en_layer = Dense(outs1, 2)
de_layer = Dense(2, outs1)

function circ(x)
    length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
    x./sqrt(sum(x .* x))
end

model = Chain(en_layer, circ, de_layer)

#CYCLOPS_FluxAutoEncoderModule.makeautoencoder_naive(outs1, n_circs, lin, lin_dim)

#=
# The below is where the gradient would be plugged in for us to use a custom gradient. Specifically, it would be everything that comes after the ->.

circ(x::TrackedArray) = track(circ, x)
Tracker.@grad function circ(x)
    return circ(Tracker.data(x)), Δ -> (Δ
=#

loss(x)= Flux.mse(model(x), x)

lossrecord = CYCLOPS_TrainingModule.@myepochs 1000 CYCLOPS_TrainingModule.mytrain!(loss, Flux.params(model), zip(norm_seed_data2), Momentum())

# This code can be uncommented or commented in order to toggle the graphing of the loss over the epochs of training that have been done and you can change the parameters of the array to focus in on some component of the graph.
#= close()
plot(lossrecord[10:200])
gcf() =#

extractmodel = Chain(en_layer, circ)

estimated_phaselist = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data1, extractmodel, n_circs)

estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)

estimated_phaselist1 = estimated_phaselist[timestamped_samples]

shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1, truetimes, "hours")


# This code replicates the first figure in the paper.
#close()
scatter(truetimes, shiftephaselist, alpha=.75, s=14)
title("Eigengenes Encoded by Single Phase")
ylabp=[0, pi/2,pi, 3*pi/2, 2*pi]
ylabs=[0, "", "π", "", "2π"]
xlabp=[0, 6, 12, 18, 24]
xlabs=["0", "6", "12", "18", "24"]
ylabel("CYCLOPS Phase (radians)", fontsize=14)
xlabel("Hour of Death", fontsize=14)
xticks(xlabp, xlabs)
yticks(ylabp, ylabs)
suptitle("CYCLOPS Phase Prediction: Human Frontal Cortex", fontsize=18)
gcf()



# The below prints to the console the relevent error statistics using the true times that we know.
errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist)
hrerrors = (12/pi) * abs.(errors)
println("Error from true times: ")
println(string("Mean: ", mean(hrerrors)))
println(string("Median: ", median(hrerrors)))
println(string("Standard Deviation: ", sqrt(var(hrerrors))))
println(string("75th percentile: ", sort(hrerrors)[Integer(round(.75 * length(hrerrors)))]))
