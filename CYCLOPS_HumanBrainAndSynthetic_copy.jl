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
# Random.seed!(12345)

# Original Data
fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
cycling = findall(in(homologue_symbol_list), fullnonseed_data[3:end,2])
fullnonseed_data_syn = CYCLOPS_PrePostProcessModule.makefloat!(fullnonseed_data[3:end,4:end])
B = trues(size(fullnonseed_data, 2))
B[2] = false
B[3] = false
fortimes = fullnonseed_data[B]
alldata_times = join(fullnonseed_data, fortimes, on = :Column1, makeunique = true)[2, 4:end] # this is just a quick fix since fullnonseed_data and fullnonseed_data_syn have the same time points in the same order.
alldata_probes = fullnonseed_data[3:end, 1] # String
alldata_symbols = fullnonseed_data[3:end, 2] # String
fullnonseed_data = CYCLOPS_PrePostProcessModule.makefloat!(fullnonseed_data[3:end,4:end])

# Synthetic Data generation
SF = 2
offset = 0.2
# .* rand(size(fullnonseed_data_syn, 1))
fullnonseed_data_syn = SF .* rand(size(fullnonseed_data_syn, 1)) .* fullnonseed_data_syn .+ mean(fullnonseed_data_syn, dims = 2) .* offset .* rand(size(fullnonseed_data_syn,1))
alldata_data = [fullnonseed_data fullnonseed_data_syn]

#= Old method
B = trues(size(fullnonseed_data_syn, 2))
B[2] = false
B[3] = false
fullnonseed_data_syn = fullnonseed_data_syn[B]
=#

# alldata_subjects = fullnonseed_data_joined[1, 4:end]
# first get the head of the dataframe which has the samples as array of Strings and then extract from that array only the headers that actually correspond with samples and not just the other headers that are there
# alldata_samples = String.(names(fullnonseed_data_joined))[4:end]

# alldata_data = fullnonseed_data_joined[3:end, 4:end]
# CYCLOPS_PrePostProcessModule.makefloat!(alldata_data) # makefloat! function no longer works.
# alldata_data = convert(Matrix, alldata_data)
# alldata_data = CYCLOPS_PrePostProcessModule.makefloat!(alldata_data)
#alldata_data = Array{Float64,2}(alldata_data)

n_samples = length(alldata_times)

timestamped_samples = setdiff(1:n_samples, findNAtime(alldata_times))

alldata_times = (Vector(alldata_times))

truetimes = mod.(Array{Float64}(alldata_times[timestamped_samples]), 24)

n_probes = length(alldata_probes)
cutrank = n_probes - MaxSeeds

Seed_MinMean = (sort(vec(mean(alldata_data, dims = 2))))[cutrank] #= Note that this number
is slightly different in new version versus old (42.88460199564358 here versus
42.88460199555892 in the old file). This is due to the fact that when the data is imported
from the CSV it is automatically rounded after a certain number of decimal points. =#

#= This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets =#
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(alldata_data, homologue_symbol_list, alldata_symbols, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, 0.97, 0.005, 30)

outs1 = outs1 - 1
norm_seed_data1 = norm_seed_data1[2:end,:]

n_batches = 2
batchsize_1 = size(norm_seed_data1, 2)
halfones = ones(trunc(Int, (batchsize_1 / 2)))
halfzeros = zeros(trunc(Int, (batchsize_1 / 2)))
norm_seed_data2 = vcat(norm_seed_data1, vcat(halfones, halfzeros)', vcat(halfzeros, halfones)')
outs2 = size(norm_seed_data2, 1)

#= Data passed into Flux models must be in the form of an array of arrays where both the
inner and outer arrays are one dimensional. This makes the array into an array of arrays. =#
norm_seed_data3 = mapslices(x -> [x], norm_seed_data2, dims=1)[:]

#= This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle =#
n_circs = 1  # set the number of circular layers in bottleneck layer
lin = false  # set the number of linear layers in bottleneck layer
lin_dim = 1  # set the in&out dimensions of the linear layers in bottleneck layer


en_layer1 = Dense(outs2, outs1)
en_layer2 = Dense(outs1, 2)
de_layer1 = Dense(2, outs1)
de_layer2 = Dense(outs2, outs1)


function circ(x)
    length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
    x./sqrt(sum(x .* x))
end


skipmatrix = zeros(outs2, n_batches)
skipmatrix[outs1 + 1, 1] = 1
skipmatrix[end, 2] = 1
skiplayer(x) = skipmatrix'*x


# PREVIOUS MODEL
Oldmodel = Chain(x -> cat(Chain(en_layer1, en_layer2, circ, de_layer1)(x), skiplayer(x); dims = 1), de_layer2)

# NEW MODEL
struct CYCLOPS
    S1  # Scaling factor for OH (encoding). Could be initialized as all ones.
    b1  # Bias factor for OH (encoding). Should be initialized as random numbers around 0.
    L1  # First linear layer (Dense). Reduced to at least 2 layers for the circ layer but can be reduced to only 3 to add one linear/non-linear layer.
    C   # Circular layer (circ(x))
    L2  # Second linear layer (Dense). Takes output from circ and any additional linear layers and expands to number of eigengenes
    S2  # Scaling factor for OH (decoding). Could be initialized as all ones
    b2  # Bias factor for OH (decoding). Should be initialized as random number around 0.
    i   # input dimension (in)
    o   # output dimensions (out)
end

wInit = (12/pi + 24/pi)/2

CYCLOPS(in::Integer, out::Integer) = CYCLOPS(param(wInit .+ 0.1 .* randn(out,in-out)), param(1 .+ rand(out,in-out)), Dense(out,2), circ, Dense(2,out), param(wInit .+ 0.1 .* randn(out,in-out)), param(1 .+ rand(out,in-out)), in, out)

function (m::CYCLOPS)(x)
    SparseOut = (x[1:m.o].*(m.S1*x[m.o + 1:end]) + m.b1*x[m.o + 1:end])
    DenseOut = m.L1(SparseOut)
    CircOut = m.C(DenseOut)
    Dense2Out = m.L2(CircOut)
    SparseOut = tanh.(Dense2Out.*(m.S2*x[m.o + 1:end]) + m.b2*x[m.o + 1:end])
end

m = CYCLOPS(outs2, outs1)

#CYCLOPS_FluxAutoEncoderModule.makeautoencoder_naive(outs1, n_circs, lin, lin_dim)

#=
# The below is where the gradient would be plugged in for us to use a custom gradient. Specifically, it would be everything that comes after the ->.

circ(x::TrackedArray) = track(circ, x)
Tracker.@grad function circ(x)
    return circ(Tracker.data(x)), Δ -> (Δ
=#

#OLD loss
Oldloss(x) = Flux.mse(Oldmodel(x), x[1:outs1])
#NEW loss
loss(x)= Flux.mse(m(x), x[1:outs1])

#OLD loss record
# Oldlossrecord = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(Oldloss, Flux.params((Oldmodel, en_layer1, en_layer2, de_layer1, de_layer2)), zip(norm_seed_data3), Momentum())
#NEW loss record
lossrecord = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss, Flux.params(m, m.S1, m.b1, m.L1, m.L2, m.S2, m.b2), zip(norm_seed_data3), Momentum())

# This code can be uncommented or commented in order to toggle the graphing of the loss over the epochs of training that have been done and you can change the parameters of the array to focus in on some component of the graph.
#=close()
plot(lossrecord[10:200])
gcf()=#
sparse(x) = (x[1:m.o].*(m.S1*x[m.i-1:end]) + m.b1*x[m.i-1:end])
Lin(x) = m.L1(x)
extractmodel = Chain(sparse, Lin, circ)
# extractOldmodel = Chain(en_layer1, en_layer2, circ)

#NEW
estimated_phaselist = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel, n_circs)
#OLD
# estimated_phaselistOld = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractOldmodel, n_circs)

#NEW
estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
#OLD
# estimated_phaselistOld = mod.(estimated_phaselistOld .+ 2*pi, 2*pi)

#NEW
estimated_phaselist1 = estimated_phaselist[timestamped_samples]
#OLD
# estimated_phaselist1Old = estimated_phaselistOld[timestamped_samples]

#NEW
shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1, truetimes, "hours")
#OLD
# shiftephaselistOld = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1Old, truetimes, "hours")

# The below prints to the console the relevent error statistics using the true times that we know.
errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist)
hrerrors = (12/pi) * abs.(errors)
# Olderrors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselistOld)
# Oldhrerrors = (12/pi) * abs.(Olderrors)
#=
println("Error from true times for New vs Old Model: ")
println(string("Mean: ", mean(hrerrors)), " vs. ", mean(Oldhrerrors))
println(string("Median: ", median(hrerrors)), " vs. ", mean(Oldhrerrors))
println(string("Standard Deviation: ", sqrt(var(hrerrors)), " vs. ", sqrt(var(Oldhrerrors))))
println(string("75th percentile: ", sort(hrerrors)[Integer(round(.75 * length(hrerrors)))], " vs. ", sort(Oldhrerrors)[Integer(round(.75 * length(Oldhrerrors)))]))
=#

# This code replicates the first figure in the paper.
#close()

# NEW MODEL FULL GRAPH
# scatter(truetimes, shiftephaselist, alpha=.75, s=14)
# OLD MODEL FULL GRAPH
# scatter(truetimes, shiftephaselistOld, alpha=.75, s=14)

# NEW MODEL FIRTS AND SECOND HALF
close()
println(string("Batch 1 mean: ", mean(hrerrors[1:convert(Int64,size(hrerrors,1)/2),1]), " vs. Batch 2 mean: ", mean(hrerrors[convert(Int64,size(hrerrors,1)/2+1):end,1])))
println(string("Batch 1 std: ", sqrt(var(hrerrors[1:convert(Int64,size(hrerrors,1)/2),1])), "vs. Batch 2 std: ", sqrt(var(hrerrors[convert(Int64,size(hrerrors,1)/2+1):end,1]))))
Batch1 = shiftephaselist[1:convert(Int64,size(shiftephaselist,1)/2),1]
Batch2 = shiftephaselist[convert(Int64,size(shiftephaselist,1)/2 + 1):end,1]
scatter(truetimes[1:convert(Int64,size(truetimes,1)/2),1], Batch1, alpha=.75, s=14)
scatter(truetimes[convert(Int64,size(truetimes,1)/2 + 1):end,1], Batch2, alpha=.75, s=14)
suptitle(string("Batch1 vs Batch2"), fontsize=18)
title(string("Scaling factor ", SF, " and offset ", offset, ". ", "Initial added to weight: ", wInit))
ylabp=[0, pi/2,pi, 3*pi/2, 2*pi]
ylabs=[0, "", "π", "", "2π"]
xlabp=[0, 6, 12, 18, 24]
xlabs=["0", "6", "12", "18", "24"]
ylabel("CYCLOPS Phase (radians)", fontsize=14)
xlabel("Hour of Death", fontsize=14)
xticks(xlabp, xlabs)
yticks(ylabp, ylabs)
gcf()

# OLD MODEL FRIST AND SECOND HALF
# scatter(truetimes[1:convert(Int64,size(truetimes,1)/2),1], shiftephaselistOld[1:convert(Int64,size(shiftephaselistOld,1)/2),1], alpha=.75, s=14)
# scatter(truetimes[convert(Int64,size(truetimes,1)/2 + 1):end,1], shiftephaselistOld[convert(Int64,size(shiftephaselistOld,1)/2 + 1):end,:], alpha=.75, s=14)

scatter(truetimes, shiftephaselist, alpha=.75, s=14)
scatter(truetimes, shiftephaselistOld, alpha=.75, s=14)
ylabp=[0, pi/2,pi, 3*pi/2, 2*pi]
ylabs=[0, "", "π", "", "2π"]
xlabp=[0, 6, 12, 18, 24]
xlabs=["0", "6", "12", "18", "24"]
ylabel("CYCLOPS Phase (radians)", fontsize=14)
xlabel("Hour of Death", fontsize=14)
xticks(xlabp, xlabs)
yticks(ylabp, ylabs)
gcf()

#=
errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist)
hrerrors = (12/pi) * abs.(errors)
println("Error from true times: ")
println(string("Mean: ", mean(hrerrors)))
println(string("Median: ", median(hrerrors)))
println(string("Standard Deviation: ", sqrt(var(hrerrors))))
println(string("75th percentile: ", sort(hrerrors)[Integer(round(.75 * length(hrerrors)))]))
=#
