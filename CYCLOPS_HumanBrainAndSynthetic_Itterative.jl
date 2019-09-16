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
Frac_Var = 0.85  # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = 0.03  # Set Number of Dimensions of SVD so that incremetal fraction of variance of var is at least this much
N_best = 10  # Number of random initial conditions to try for each optimization
total_background_num = 10  # Number of background runs for global background refrence distribution (bootstrap). For real runs, this should be much higher.

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

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

# Model 1 + normalrand(0.15)
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

function fillcyclops(i::Int, o::Int, w::Array{Int64,1}, l::Int=0)
    length(w) <= nthreads() || throw(ArgumentError(string("Too many initial conditions in w. Only ", nthreads(), " conditions allowed for multicore processing, but ", length(w), " given.")))
    i>(o+1) || throw(ArgumentError(string("INPUT DIMENSIONS (i) must be at least OUTPUT DIMENSIONS (o) + 2, but is only OUTPUT DIMENSIONS (o) + ",i-o,".")))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be less than 0.")))
    (o-l-1)>2 || @warn "Limited data reduction"
    ms = fill(cyclops(param(w[1] .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), Dense(o,2+l), circ, Dense(2+l,o), param(w[1] .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), i, o), length(w), 1)
    for ii in 1:length(w)
        ms[ii] = cyclops(param(w[ii] .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), Dense(o,2+l), circ, Dense(2+l,o), param(w[ii] .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), i, o)
    end
    return ms
end

function fillcyclops(i::Int, o::Int, w::Int=1, l::Int=0)
    i > (o+1) || throw(ArgumentError(string("Input dimensions must be at least output dimension + 2, but is only output dimension + ",i-o)))
    !(l<0) || throw(ArgumentError(string("Number of linear layers (l) cannot be negative.")))
    (o-l-1)>2 || @warn "Limited data reduction"
    cyclops(param(w .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), Dense(o,2+l), circ, Dense(2+l,o), param(w .+ 0.15 .* randn(o, i - o)), param(1 .+ 0.1 .* rand(o, i - o)), i, o)
end

function gensynthdata(ogdataframe::DataFrame, SF::Int, offset::Float64=0.0)
    ogdata = CYCLOPS_PrePostProcessModule.makefloat!(ogdataframe[3:end,4:end])
    syndata = (SF .+ 0.25 .* rand(size(ogdata,1))) .* ogdata .+ mean(ogdata, dims = 2) .* offset .* rand(size(ogdata,1))
    return fulldata = [ogdata syndata]
end

# Gathering/generating data
# Original Data
homologue_symbol_list = CSV.read("Human_UbiquityCyclers.csv")[1:end, 2]
fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
SF = 2 # random scaling of each row in synthetic data between 0 and SF.
offset = 0.2 # random offset for each row of synthetic data between 0 and offseet * (mean of the row)
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
truetimes = mod.(Array{Float64}(alldata_times[timestamped_samples]), 24) # list of all times, excluding NA, in 24 hours
n_probes = length(alldata_probes) # number of total probes before reduction
cutrank = n_probes - MaxSeeds # How many of the total genes are included in the list used for SVD
Seed_MinMean = (sort(vec(mean(alldata_data[:,1:202], dims = 2))))[cutrank] # finding the mean of the row of the last included gene after sorting all in descending order

# This extracts the genes from the dataset that were felt to have a high likelyhood to be cycling - and also had a reasonable coefficient of variation in this data sets
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(alldata_data, homologue_symbol_list, alldata_symbols, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, 0.97, 0.0075, 30)

# outs1 = outs1 - 1
# norm_seed_data1 = norm_seed_data1[2:end,:]

# Creating one-hot data
n_batches = 2
batchsize_1 = size(norm_seed_data1, 2)
halfones = ones(trunc(Int, (batchsize_1 / 2)))
halfzeros = zeros(trunc(Int, (batchsize_1 / 2)))
norm_seed_data2 = vcat(norm_seed_data1, vcat(halfones, halfzeros)', vcat(halfzeros, halfones)')
outs2 = size(norm_seed_data2, 1)
norm_seed_data3 = mapslices(x -> [x], norm_seed_data2, dims=1)[:] # Data passed into Flux models must be in the form of an array of arrays where both the inner and outer arrays are one dimensional. This makes the array into an array of arrays.

trainingdata = (norm_seed_data2, norm_seed_data3, truetimes, timestamped_samples)

ms = fillcyclops(outs2, outs1, w)

# This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle
n_circs = 1  # set the number of circular layers in bottleneck layer
lin = false  # set the number of linear layers in bottleneck layer
lin_dim = 1  # set the in&out dimensions of the linear layers in bottleneck layer

function doesnothing(x)
    return x
end

# td::Tuple{Array{Float64,2},Array{Array{Float64,1},1},Array{Float64,1},Array{Int64,1}}

function cyclopsfunctions(td::Tuple, ms::Array{cyclops,2}, fs::Array{Function,1}=[doesnothing], nc::Int=1)
    for ii in 1:length(fs)
        function (m::cyclops)(x)
            SparseOut = fs[ii].(x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = fs[ii].(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end
        @threads for jj in 1:length(ms)
            loss(x)= Flux.mse(ms[jj](x), x[1:outs1])
            lossrecord = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss, Flux.params(ms[jj].S1, ms[jj].b1, ms[jj].L1, ms[jj].L2, ms[jj].S2, ms[jj].b2), zip(td[2]), Momentum())
            sparse(x) = (x[1:ms[jj].o].*(ms[jj].S1*x[ms[jj].i-1:end]) + ms[jj].b1*x[ms[jj].i-1:end])
            lin(x) = ms[jj].L1(x)
            trainedmodel = Chain(sparse, Lin, circ)
            estimated_phaselist = CYCLOPS_FluxAutoEncoderModule.extractphase(td[1], trainedmodel, nc)
            estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
            estimated_phaselist = estimated_phaselist[td[4]]
            shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist, tt, "hours")
            errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * td[3] / 24, shiftephaselist)
            hrerrors = (12/pi) * abs.(errors)
            Batch1 = mean(hrerrors[1:convert(Int64,size(hrerrors,1)/2),1])
            Batch2 = mean(hrerrors[convert(Int64,size(hrerrors,1)/2+1):end,1])


for jj = 1:3

    if jj == 1 # No non-linearity
        function (m::CYCLOPS1)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = (Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

        function (m::CYCLOPS3)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = (Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

        function (m::CYCLOPS5)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = (Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

    elseif jj == 2 # Non-linearity in encoding AND decoding layer

        function (m::CYCLOPS1)(x)
            SparseOut = tanh.(x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

        function (m::CYCLOPS3)(x)
            SparseOut = tanh.(x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

        function (m::CYCLOPS5)(x)
            SparseOut = tanh.(x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end

    else # Non-linearity in decoding layer only

        function (m::CYCLOPS1)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end
        function (m::CYCLOPS3)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end
        function (m::CYCLOPS5)(x)
            SparseOut = (x[1:m.o] .* (m.S1 * x[m.o + 1:end]) + m.b1 * x[m.o + 1:end])
            DenseOut = m.L1(SparseOut)
            CircOut = m.C(DenseOut)
            Dense2Out = m.L2(CircOut)
            SparseOut = tanh.(Dense2Out .* (m.S2 * x[m.o + 1:end]) + m.b2 * x[m.o + 1:end])
        end
    end

    for kk = 1:5
        # Encoding and Decoding layers of previous generation CYCLOPS
        en_layer1 = Dense(outs2, outs1)
        en_layer2 = Dense(outs1, 2)
        de_layer1 = Dense(2, outs1)
        de_layer2 = Dense(outs2, outs1)

        # One-hot structure for previous generation CYCLOPS
        skipmatrix = zeros(outs2, n_batches)
        skipmatrix[outs1 + 1, 1] = 1
        skipmatrix[end, 2] = 1
        skiplayer(x) = skipmatrix'*x


        # Previous generation CYCLOPS
        Oldmodel = Chain(x -> cat(Chain(en_layer1, en_layer2, circ, de_layer1)(x), skiplayer(x); dims = 1), de_layer2)

        # New CYCLOPS
        m1 = CYCLOPS1(outs2, outs1)
        m3 = CYCLOPS3(outs2, outs1)
        m5 = CYCLOPS5(outs2, outs1)

        # OLD loss
        # Oldloss(x) = Flux.mse(Oldmodel(x), x[1:outs1])
        # NEW loss
        loss1(x)= Flux.mse(m1(x), x[1:outs1])
        loss3(x)= Flux.mse(m3(x), x[1:outs1])
        loss5(x)= Flux.mse(m5(x), x[1:outs1])

        # Previous loss record
        # Oldlossrecord = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(Oldloss, Flux.params((Oldmodel, en_layer1, en_layer2, de_layer1, de_layer2)), zip(norm_seed_data3), Momentum())
        # New loss record
        lossrecord1 = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss1, Flux.params(m1.S1, m1.b1, m1.L1, m1.L2, m1.S2, m1.b2), zip(norm_seed_data3), Momentum())
        lossrecord3 = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss1, Flux.params(m3.S1, m3.b1, m3.L1, m3.L2, m3.S2, m3.b2), zip(norm_seed_data3), Momentum())
        lossrecord5 = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss1, Flux.params(m5.S1, m5.b1, m5.L1, m5.L2, m5.S2, m5.b2), zip(norm_seed_data3), Momentum())


        sparse1(x) = (x[1:m1.o].*(m1.S1*x[m1.i-1:end]) + m1.b1*x[m1.i-1:end])
        sparse3(x) = (x[1:m3.o].*(m3.S1*x[m3.i-1:end]) + m3.b1*x[m3.i-1:end])
        sparse5(x) = (x[1:m5.o].*(m5.S1*x[m5.i-1:end]) + m5.b1*x[m5.i-1:end])
        Lin1(x) = m1.L1(x)
        Lin3(x) = m3.L1(x)
        Lin5(x) = m5.L1(x)
        extractmodel1 = Chain(sparse1, Lin1, circ)
        extractmodel3 = Chain(sparse3, Lin3, circ)
        extractmodel5 = Chain(sparse5, Lin5, circ)
        # extractOldmodel = Chain(en_layer1, en_layer2, circ)

        #NEW
        estimated_phaselist1 = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel1, n_circs)
        estimated_phaselist1 = mod.(estimated_phaselist1 .+ 2*pi, 2*pi)
        estimated_phaselist1 = estimated_phaselist1[timestamped_samples]
        shiftephaselist1 = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1, truetimes, "hours")
        estimated_phaselist3 = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel3, n_circs)
        estimated_phaselist3 = mod.(estimated_phaselist3 .+ 2*pi, 2*pi)
        estimated_phaselist3 = estimated_phaselist3[timestamped_samples]
        shiftephaselist3 = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist3, truetimes, "hours")
        estimated_phaselist5 = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel5, n_circs)
        estimated_phaselist5 = mod.(estimated_phaselist5 .+ 2*pi, 2*pi)
        estimated_phaselist5 = estimated_phaselist5[timestamped_samples]
        shiftephaselist5 = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist5, truetimes, "hours")
        #OLD
        # estimated_phaselistOld = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractOldmodel, n_circs)
        # estimated_phaselistOld = mod.(estimated_phaselistOld .+ 2*pi, 2*pi)
        # estimated_phaselist1Old = estimated_phaselistOld[timestamped_samples]

        #NEW

        #OLD
        # shiftephaselistOld = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1Old, truetimes, "hours")

        # The below prints to the console the relevent error statistics using the true times that we know.
        errors1 = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist1)
        errors3 = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist3)
        errors5 = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist5)
        hrerrors1 = (12/pi) * abs.(errors1)
        hrerrors3 = (12/pi) * abs.(errors3)
        hrerrors5 = (12/pi) * abs.(errors5)
        # Olderrors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselistOld)
        # Oldhrerrors = (12/pi) * abs.(Olderrors)

        # NEW MODEL FIRTS AND SECOND HALF
        close()
        println(string("Batch 1 mean: ", mean(hrerrors[1:convert(Int64,size(hrerrors,1)/2),1]), " vs. Batch 2 mean: ", mean(hrerrors[convert(Int64,size(hrerrors,1)/2+1):end,1])))
        println(string("Batch 1 std: ", sqrt(var(hrerrors[1:convert(Int64,size(hrerrors,1)/2),1])), "vs. Batch 2 std: ", sqrt(var(hrerrors[convert(Int64,size(hrerrors,1)/2+1):end,1]))))
        Batch1 = shiftephaselist[1:convert(Int64,size(shiftephaselist,1)/2),1]
        Batch2 = shiftephaselist[convert(Int64,size(shiftephaselist,1)/2 + 1):end,1]
        scatter(truetimes[1:convert(Int64,size(truetimes,1)/2),1], Batch1, alpha=.75, s=14)
        scatter(truetimes[convert(Int64,size(truetimes,1)/2 + 1):end,1], Batch2, alpha=.75, s=14)
        suptitle(string("New Model: Batch1 (B) vs Batch2 (O)"), fontsize=18)
        title(string("Scaling factor ", SF, " and offset ", offset, ". ", "Weights init. at ", wInit, " + 0 - 0.1"))
        ylabp=[0, pi/2,pi, 3*pi/2, 2*pi]
        ylabs=[0, "", "π", "", "2π"]
        xlabp=[0, 6, 12, 18, 24]
        xlabs=["0", "6", "12", "18", "24"]
        ylabel("CYCLOPS Phase (radians)", fontsize=14)
        xlabel("Hour of Death", fontsize=14)
        xticks(xlabp, xlabs)
        yticks(ylabp, ylabs)
        gcf()
        # if !isdir(base * "/Downloads/Research")
        # savefig()
        close()
        # OLD MODEL FRIST AND SECOND HALF
        # scatter(truetimes[1:convert(Int64,size(truetimes,1)/2),1], shiftephaselistOld[1:convert(Int64,size(shiftephaselistOld,1)/2),1], alpha=.75, s=14)
        # scatter(truetimes[convert(Int64,size(truetimes,1)/2 + 1):end,1], shiftephaselistOld[convert(Int64,size(shiftephaselistOld,1)/2 + 1):end,:], alpha=.75, s=14)

        # scatter(truetimes, shiftephaselist, alpha=.75, s=14)
        # scatter(truetimes, shiftephaselistOld, alpha=.75, s=14)
        # ylabp=[0, pi/2,pi, 3*pi/2, 2*pi]
        # ylabs=[0, "", "π", "", "2π"]
        # xlabp=[0, 6, 12, 18, 24]
        # xlabs=["0", "6", "12", "18", "24"]
        # ylabel("CYCLOPS Phase (radians)", fontsize=14)
        # xlabel("Hour of Death", fontsize=14)
        # xticks(xlabp, xlabs)
        # yticks(ylabp, ylabs)
        # gcf()
    end
end
