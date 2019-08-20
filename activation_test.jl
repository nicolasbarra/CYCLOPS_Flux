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

function makefloat!(ar::Array{Any}) # convert to Array{Any} first, using convert{Matrix, df}.
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float64, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float64, 2}, ar)

    ar
end

function makefloat!(df::DataFrame) # will convert to Array{Float} first
    ar = convert(Matrix, df)
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float64, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float64, 2}, ar)

    ar
end

Frac_Var = 0.85
DFrac_Var = 0.03
N_best = 10
total_background_num = 10

homologue_symbol_list = CSV.read("Human_UbiquityCyclers.csv")[1:end, 2]

Seed_MinCV = 0.14
Seed_MaxCV = .7
Seed_Blunt =.975
MaxSeeds = 10000

Random.seed!(12345)

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
fullnonseed_data_syn = CSV.read("Annotated_Unlogged_BA11Data_r15_v1.csv")
B = trues(size(fullnonseed_data_syn, 2))
B[2] = false
B[3] = false
fullnonseed_data_syn = fullnonseed_data_syn[B]
fullnonseed_data_joined = join(fullnonseed_data, fullnonseed_data_syn, on = :Column1, makeunique = true)
alldata_probes = fullnonseed_data_joined[3:end, 1]
alldata_symbols = fullnonseed_data_joined[3:end, 2]
alldata_subjects = fullnonseed_data_joined[1, 4:end]
alldata_times = fullnonseed_data_joined[2, 4:end]

alldata_samples = String.(names(fullnonseed_data_joined))[4:end]

alldata_data = fullnonseed_data_joined[3:end, 4:end]
alldata_data = makefloat!(alldata_data)

n_samples = length(alldata_times)

timestamped_samples = setdiff(1:n_samples, findNAtime(alldata_times))

alldata_times = (Vector(alldata_times))

truetimes = mod.(Array{Float64}(alldata_times[timestamped_samples]), 24)

n_probes = length(alldata_probes)
cutrank = n_probes - MaxSeeds

Seed_MinMean = (sort(vec(mean(alldata_data, dims = 2))))[cutrank]
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(alldata_data, homologue_symbol_list, alldata_symbols, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, Frac_Var, DFrac_Var, 30)

#= TODO Fix one hot matrix generation. Hard coded right now. =#
n_batches = 2
batchsize_1 = size(norm_seed_data1, 2)
halfones = ones(trunc(Int, (batchsize_1 / 2)))
halfzeros = zeros(trunc(Int, (batchsize_1 / 2)))
norm_seed_data2 = vcat(norm_seed_data1, vcat(halfones, halfzeros)', vcat(halfzeros, halfones)')
outs2 = size(norm_seed_data2, 1)

norm_seed_data3 = mapslices(x -> [x], norm_seed_data2, dims=1)[:]

n_circs = 1
lin = false
lin_dim = 1

en_layer1 = Dense(outs2, outs1, )
en_layer2 = Dense(outs1, 2)
de_layer1 = Dense(2, outs1)
de_layer2 = Dense(outs2, outs1)

en2_layer1 = Dense(outs2, outs1, atan)
en2_layer2 = Dense(outs1, 2)
de2_layer1 = Dense(2, outs1)
de2_layer2 = Dense(outs2, outs1,atan)

function circ(x)
    length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
    x./sqrt(sum(x .* x))
end

skipmatrix = zeros(outs2, n_batches)
skipmatrix[6, 1] = 1
skipmatrix[7, 2] = 1
skiplayer(x) = skipmatrix'*x

model = Chain(x -> cat(Chain(en_layer1, en_layer2, circ, de_layer1)(x), skiplayer(x); dims = 1), de_layer2)
model2 = Chain(x -> cat(Chain(en2_layer1, en2_layer2, circ, de2_layer1)(x), skiplayer(x); dims = 1), de2_layer2)

loss(x)= Flux.mse(model(x), x[1:outs1])
loss2(x)= Flux.mse(model2(x), x[1:outs1])

lossrecord = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss, Flux.params((model, en_layer1, en_layer2, de_layer1, de_layer2)), zip(norm_seed_data3), Momentum())
lossrecord2 = CYCLOPS_TrainingModule.@myepochs 750 CYCLOPS_TrainingModule.mytrain!(loss2, Flux.params((model2, en2_layer1, en2_layer2, de2_layer1, de2_layer2)), zip(norm_seed_data3), Momentum())

#=close()
plot(lossrecord[10:200])
gcf()=#

extractmodel = Chain(en_layer1, en_layer2, circ)
extractmodel2 = Chain(en2_layer1, en2_layer2, circ)

estimated_phaselist = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel, n_circs)
estimated_phaselist2 = CYCLOPS_FluxAutoEncoderModule.extractphase(norm_seed_data2, extractmodel2, n_circs)

estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
estimated_phaselist2 = mod.(estimated_phaselist2 .+ 2*pi, 2*pi)

estimated_phaselist1 = estimated_phaselist[timestamped_samples]
estimated_phaselist12 = estimated_phaselist2[timestamped_samples]

shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1, truetimes, "hours")
shiftephaselist2 = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist12, truetimes, "hours")


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

scatter(truetimes, shiftephaselist2, alpha=.75, s=14)
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

errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist)
errors2 = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist2)
hrerrors = (12/pi) * abs.(errors)
hrerrors2 = (12/pi) * abs.(errors2)
println("Error from true times: ")
println(string("Mean1: ", mean(hrerrors)))
println(string("Mean2: ", mean(hrerrors2)))
println(string("Median: ", median(hrerrors)))
println(string("Median2: ", median(hrerrors2)))
println(string("Standard Deviation: ", sqrt(var(hrerrors))))
println(string("Standard Deviation2: ", sqrt(var(hrerrors2))))
println(string("75th percentile: ", sort(hrerrors)[Integer(round(.75 * length(hrerrors)))]))
println(string("75th percentile2: ", sort(hrerrors2)[Integer(round(.75 * length(hrerrors2)))]))
