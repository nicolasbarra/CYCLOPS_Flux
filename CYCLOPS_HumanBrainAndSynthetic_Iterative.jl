using Flux, CSV, Statistics, Distributed, Juno, PyPlot, BSON, Revise, DataFrames, Base.Threads

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

# seed Random Number Generator for reproducible results
Random.seed!(12345)

# Reduction information
Frac_Var = 0.97  # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = 0.0075  # Set Number of Dimensions of SVD so that incremetal fraction of variance of var is at least this much
N_best = 10  # Number of random initial conditions to try for each optimization
total_background_num = 10  # Number of background runs for global background refrence distribution (bootstrap). For real runs, this should be much higher.

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

ops = [Frac_Var, DFrac_Var, Seed_MinCV, Seed_MaxCV, Seed_Blunt, MaxSeeds]

w = [1, 3, 5, 7]

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

function circ(x)
    length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
    x./sqrt(sum(x .* x))
end

function doesnothing(x) # pseudo function that does nothing to x, used as placeholder function for cyclops
    return x
end

# Model 1
struct cyclops
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

function gencyclops(i::Int, o::Int, w::Array{<:Real}, l::Int=0, nc::Int=1)
    i>(o+1) || throw(ArgumentError(string("INPUT DIMENSIONS (i) must be at least OUTPUT DIMENSIONS (o) + 2, but is OUTPUT DIMENSIONS (o) + ",i-o,".")))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be less than 0.")))
    (o-l)>3 || @warn "Limited data reduction"
    S1 = Array{Float32,2}(w[1] .+ 0.15 .* randn(o, i - o))
    b1 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S2 = Array{Float32,2}(w[1] .+ 0.15 .* randn(o, i - o))
    b2 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S1[1,2] = -S1[1,2]
    S2[1,2] = -S2[1,2]
    ms = fill(cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o), length(w), 1)
    for ii in 2:length(w)
        S1 = Array{Float32,2}(w[ii] .+ 0.15 .* randn(o, i - o))
        b1 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
        S2 = Array{Float32,2}(w[ii] .+ 0.15 .* randn(o, i - o))
        b2 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
        S1[1,2] = -S1[1,2]
        S2[1,2] = -S2[1,2]
        ms[ii] = cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o)
    end

    ms
end

function gencyclops(i::Int, o::Int, w::Real=1, l::Int=0, nc::Int=1)
    i > (o+1) || throw(ArgumentError(string("Input dimensions must be at least output dimension + 2, but is only output dimension + ",i-o)))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be negative.")))
    (o-l)>3 || @warn "Limited data reduction"
    S1 = Array{Float32,2}(w .+ 0.15 .* randn(o, i - o))
    b1 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S2 = Array{Float32,2}(w .+ 0.15 .* randn(o, i - o))
    b2 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S1[1,2] = -S1[1,2]
    S2[1,2] = -S2[1,2]
    [cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o)]
end

function gencyclopsb(i::Int, o::Int, b::Array{<:Real}, l::Int=0, nc::Int=1)
    i>(o+1) || throw(ArgumentError(string("INPUT DIMENSIONS (i) must be at least OUTPUT DIMENSIONS (o) + 2, but is OUTPUT DIMENSIONS (o) + ",i-o,".")))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be less than 0.")))
    (o-l)>3 || @warn "Limited data reduction"
    S1 = Array{Float32,2}(1 .+ 0.15 .* randn(o, i - o))
    b1 = Array{Float32,2}(b[1] .+ 0.1 .* rand(o, i - o))
    S2 = Array{Float32,2}(1 .+ 0.15 .* randn(o, i - o))
    b2 = Array{Float32,2}(b[1] .+ 0.1 .* rand(o, i - o))
    b1[1,2] = -S1[1,2]
    S2[1,2] = -S2[1,2]
    ms = fill(cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o), length(w), 1)
    for ii in 2:length(w)
        S1 = Array{Float32,2}(1 .+ 0.15 .* randn(o, i - o))
        b1 = Array{Float32,2}(b[ii] .+ 0.1 .* rand(o, i - o))
        S2 = Array{Float32,2}(1 .+ 0.15 .* randn(o, i - o))
        b2 = Array{Float32,2}(b[ii] .+ 0.1 .* rand(o, i - o))
        S1[1,2] = -S1[1,2]
        S2[1,2] = -S2[1,2]
        ms[ii] = cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o)
    end

    ms
end

function gencyclopsb(i::Int, o::Int, w::Real=1, l::Int=0, nc::Int=1)
    i > (o+1) || throw(ArgumentError(string("Input dimensions must be at least output dimension + 2, but is only output dimension + ",i-o)))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be negative.")))
    (o-l)>3 || @warn "Limited data reduction"
    S1 = Array{Float32,2}(w .+ 0.15 .* randn(o, i - o))
    b1 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S2 = Array{Float32,2}(w .+ 0.15 .* randn(o, i - o))
    b2 = Array{Float32,2}(1 .+ 0.1 .* rand(o, i - o))
    S1[1,2] = -S1[1,2]
    S2[1,2] = -S2[1,2]
    [cyclops(param(S1), param(b1), Dense(o,2+l), circ, Dense(2+l,o), param(S2), param(b2), i, o)]
end

function gensynthdata(ogdataframe::DataFrame, SF::Real, offset::Real=0.0)
    center = (1 + SF)/2 # mean is half way between 1 and SF
    stdev = (SF - 1)/(2*1.96) # stdev is such that 2 stdev lie between 1 and SF
    ogdata = CYCLOPS_PrePostProcessModule.makefloat!(ogdataframe[3:end,4:end]) # convert data to matrix of Float32
    syndata = (center .+ stdev .* rand(size(ogdata,1))) .* ogdata .+ (mean(ogdata, dims = 2) .* offset .* rand(size(ogdata,1))) # add scaling factor and percent of mean offset to create synth data

    [ogdata syndata] # new full data set
end

function (m::cyclops)(x)
    SparseOut = tanh.(x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end]) # OH layer from input
    DenseOut = m.L1(SparseOut) # Fully conneted layer between OH and circ
    CircOut = m.C(DenseOut) # circular layer
    Dense2Out = m.L2(CircOut) # Full connected layer between circ and OH
    return Sparse2Out = (Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end]) # OH layer to output
end

function trainmetrics(td, ms, tms, ferrors, ii::Int, jj::Int, epochs::Int=100, nc=1)
    m = ms[jj]
    loss(x)= Flux.mse(m(x), x[1:td[2]]) # define loss function used for training
    lossrecord = CYCLOPS_TrainingModule.@myepochs 100 CYCLOPS_TrainingModule.mytrain!(loss, Flux.params(m.S1, m.b1, m.L1, m.L2, m.S2, m.b2), zip(td[end-2]), Momentum())
    sparse(x) = (x[1:m.o].*(m.S1*x[m.i-1:end]) + m.b1*x[m.i-1:end])
    lin(x) = m.L1(x)
    trainedmodel = Chain(sparse, lin, circ)
    estimated_phaselist = CYCLOPS_FluxAutoEncoderModule.extractphase(td[3], trainedmodel, nc)
    estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
    estimated_phaselist = estimated_phaselist[td[end]]
    shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist, td[end-1], "hours")
    errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * td[end-1] / 24, shiftephaselist)
    hrerrors = (12/pi) * abs.(errors)
    Batch1 = mean(hrerrors[1:Int(size(hrerrors,1)/2),1])
    Batch2 = mean(hrerrors[Int(size(hrerrors,1)/2+1):end,1])
    tms[jj,ii] = m
    ferrors[jj,ii] = [Batch1, Batch2]

    return tms, ferrors
end

function multimodeltrain(td, w=1, reps::Int=5, nc::Int=1)

    tms = fill(gencyclops(td[1],td[2],0)[1],size(w,1),reps) # Array{cyclops,2} that will contain all trained models
    ferrors = fill(Array{Float32,1}([Inf, Inf]), size(w,1), reps) # Array{Float32,2} that will contain all errors for batch 1 and batch 2

    for ii = 1:reps # create new models with scaling weights centered around w five times

        ms = gencyclops(td[1], td[2], w) # using fillcyclops to generate an array of models

        for jj = 1:size(ms,1) # train one model on each core. The number of models that can be trained at once depend on the number of weights

            tms, ferrors = trainmetrics(td, ms, tms, ferrors, ii, jj)
        end
    end

    tms, ferrors
end

function gentrainingdata(ogdata::String, hsl::String, SF::Number, offset::Number, ops)

    # Gathering/generating data
    # Original Data
    homologue_symbol_list = CSV.read(hsl)[1:end, 2]
    fullnonseed_data = CSV.read(ogdata)
    # SF = random scaling of each row in synthetic data, such that that 95% of the rows are scaled between 1 and SF. (normal random)
    # offset = random offset for each row of synthetic data between 0 and offseet * (mean of the row). (uniform random)
    alldata_data = gensynthdata(fullnonseed_data, SF, offset) # full data used for training

    # cycling = findall(in(homologue_symbol_list), fullnonseed_data[3:end,2])
    B = trues(size(fullnonseed_data, 2))
    B[2] = false
    B[3] = false
    fortimes = fullnonseed_data[B]
    alldata_times = join(fullnonseed_data, fortimes, on = :Column1, makeunique = true)[2, 4:end] # this is just a quick fix since fullnonseed_data and fullnonseed_data_syn have the same time points in the same order.
    alldata_probes = fullnonseed_data[3:end, 1] # String
    alldata_symbols = fullnonseed_data[3:end, 2] # String

    # Data dimension information
    n_samples = length(alldata_times) # all samples
    timestamped_samples = setdiff(1:n_samples, findNAtime(alldata_times)) # find all samples with true time of death recorded
    alldata_times = (Vector(alldata_times)) # list of all times, including NA
    truetimes = mod.(Array{Float32}(alldata_times[timestamped_samples]), 24) # list of all times, excluding NA, in 24 hours
    n_probes = length(alldata_probes) # number of total probes before reduction
    cutrank = Int(n_probes - ops[end]) # How many of the total genes are included in the list used for SVD
    Seed_MinMean = (sort(vec(mean(alldata_data, dims = 2))))[cutrank] # finding the mean of the row of the last included gene after sorting all in descending order

    # This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets
    seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(alldata_data, homologue_symbol_list, alldata_symbols, ops[4], ops[3], Seed_MinMean, ops[end-1])
    seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
    outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, ops[1], ops[2], 30)

    # outs1 = outs1 - 1
    # norm_seed_data1 = norm_seed_data1[2:end,:]

    # Creating one-hot data
    n_batches = 2
    batchsize_1 = size(norm_seed_data1, 2)
    halfones = Array{Float32}(ones(trunc(Int, (batchsize_1 / 2))))
    halfzeros = Array{Float32}(zeros(trunc(Int, (batchsize_1 / 2))))
    norm_seed_data2 = vcat(norm_seed_data1, vcat(halfones, halfzeros)', vcat(halfzeros, halfones)')
    outs2 = size(norm_seed_data2, 1)
    norm_seed_data3 = mapslices(x -> [x], norm_seed_data2, dims=1)[:] # Data passed into Flux models must be in the form of an array of arrays where both the inner and outer arrays are one dimensional. This makes the array into an array of arrays.

    return td = (outs2, outs1, norm_seed_data2, norm_seed_data3, truetimes, timestamped_samples) # later becomes td in function
end

# This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle
n_circs = 1  # set the number of circular layers in bottleneck layer
lin = false  # set the number of linear layers in bottleneck layer
lin_dim = 1  # set the in&out dimensions of the linear layers in bottleneck layer


ogdata = "Annotated_Unlogged_BA11Data.csv"
hsl = "Human_UbiquityCyclers.csv"

ops = [Frac_Var, DFrac_Var, Seed_MinCV, Seed_MaxCV, Seed_Blunt, MaxSeeds]
SF = 2
offset = 0.2
td = gentrainingdata(ogdata, hsl, SF, offset, ops)

tms, ferrors = multimodeltrain(td,w)
