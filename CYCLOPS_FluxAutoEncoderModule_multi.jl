using Flux, CSV, Statistics, Distributed, Juno, PyPlot, BSON

import Random

@everywhere basedir = homedir()
@everywhere cd(basedir * "/github/CYCLOPS_Flux")
@everywhere include("CYCLOPS_CircularStatsModule.jl")
@everywhere include("CYCLOPS_PrePostProcessModule.jl")
@everywhere include("CYCLOPS_SeedModule.jl")
@everywhere include("CYCLOPS_SmoothModule_multi.jl")
@everywhere include("CYCLOPS_MyTrainSuppportModule.jl")

#= make all the columns (beginning at inputted column number) of a the DataFrame of type
Float64, not String since they are Numbers =#
function makefloat!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) == Array{String,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end

# find the samples that have no time stamp so you can remove them
function findNAtime(df)
  r = []
  for row in 1:length(df)
    if typeof(df[row]) == String
      append!(r, row)
    end
  end

  r
end

# extracts the phase angles from the model for analysis
function extractphase(data_matrix, model)
    points = size(data_matrix, 2)
    phases = zeros(points)
    for n in 1:points
        phases[n] = Tracker.data(atan(model[1:2](data_matrix[:, n])[2], model[1:2](data_matrix[:, n])[1]))
    end

    phases
end

Frac_Var = 0.85  # Set Number of Dimensions of SVD to maintain this fraction of variance
DFrac_Var = 0.03  #= Set Number of Dimensions of SVD so that incremetal fraction of
variance of var is at least this much=#
N_best = 10  # Number of random initial conditions to try for each optimization
total_background_num = 10  #= Number of background runs for global background refrence
distribution (bootstrap). For real runs, this should be much higher. =#

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
# first get the head of the dataframe which has the samples as array of Strings
alldata_samples = String.(names(fullnonseed_data))
#= then extract from that array only the headers that actually correspond with samples and
not just the other headers that are there =#
alldata_samples = alldata_samples[4:end]

alldata_data = fullnonseed_data[3:end, 4:end]
makefloat!(1, alldata_data)
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
seed_symbols1, seed_data1 = CYCLOPS_SeedModule.getseed_homologuesymbol_brain(fullnonseed_data, homologue_symbol_list1, Seed_MaxCV, Seed_MinCV, Seed_MinMean, Seed_Blunt)
seed_data1 = CYCLOPS_SeedModule.dispersion!(seed_data1)
outs1, norm_seed_data1 = CYCLOPS_PrePostProcessModule.getEigengenes(seed_data1, Frac_Var, DFrac_Var, 30)

# TODO: Figure out why this is needed below
#=
outs1 = 5
norm_seed_data1 = norm_seed_data1[1:5, :]
=#

#= Data passed into Flux models must be in the form of an array of arrays where both the
inner and outer arrays are one dimensional. This makes the array into an array of arrays. =#
norm_seed_data2 = mapslices(x -> [x], norm_seed_data1, dims=1)[:]

#= This example creates a "balanced autoencoder" where the eigengenes ~ principle components are encoded by a single phase angle =#
encoder = Dense(outs1, 2)
function circ(x)
  length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
  x./sqrt(sum(x .* x))
end
lin = Dense(2, 2)
decoder = Dense(4, outs1)

#=
# The below is where the gradient would be plugged in for us to use a custom gradient. Specifically, it would be everything that comes after the ->.

circ(x::TrackedArray) = track(circ, x)
Tracker.@grad function circ(x)
    return circ(Tracker.data(x)), Δ -> (Δ
=#

model = Chain(encoder, x -> cat(circ(x), lin(x); dims = 1), decoder)

#modelcomplex = Chain(encoderA1, x -> cat(bottlenecklinear(x), bottleneckcircular(x), dims=), decoder)

loss(x) = Flux.mse(model(x), x)

function mytrain!(loss, ps, data, opt; cb = () -> ())
  lossrec =[]
  ps = Flux.Params(ps)
  cb = CYCLOPS_MyTrainSuppportModule.runall(cb)
  @progress for d in data
    try
      gs = Flux.gradient(ps) do
        loss(d...)
      end
      Tracker.update!(opt, ps, gs)
      append!(lossrec, loss(d...))
      if cb() == :stop
        break
      end
    catch ex
      if ex isa CYCLOPS_MyTrainSuppportModule.StopException
        break
      else
        rethrow(ex)
      end
    end
  end
  avg = mean(lossrec)
  println(string("Average loss this epoch: ", avg))

  Tracker.data(avg)
end

macro myepochs(n, ex)
  return :(lossrecord = [];
  @progress for i = 1:$(esc(n))
      @info "Epoch $i"
      avgloss = $(esc(ex))
      if size(lossrecord, 3) > 1 && avgloss > avgloss[size(avgloss, 1)] && avgloss > avgloss[size(avgloss, 1) - 1]
        break
      else
        append!(lossrecord, avgloss)
      end
    end;

    lossrecord)
end

Flux.@epochs 1 mytrain!(loss, Flux.params(model), zip(norm_seed_data2), Descent(0.01))

# This code can be uncommented or commented in order to toggle the graphing of the loss over the epochs of training that have been done and you can change the parameters of the array to focus in on some component of the graph.
close()
plot(Tracker.data(lossrecs[200:end]))
gcf()


estimated_phaselist = extractphase(norm_seed_data1, model)
estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)

estimated_phaselist1 = estimated_phaselist[timestamped_samples]

shiftephaselist = CYCLOPS_PrePostProcessModule.best_shift_cos(estimated_phaselist1, truetimes, "hours")

#=
# This code replicates the first figure in the paper.

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
=#

#  The below prints to the console the relevent error statistics using the true times that we know.
errors = CYCLOPS_CircularStatsModule.circularerrorlist(2*pi * truetimes / 24, shiftephaselist)
hrerrors = (12/pi) * abs.(errors)
println("Error from true times: ")
println(string("Mean: ", mean(hrerrors)))
println(string("Median: ", median(hrerrors)))
println(string("Standard Deviation: ", sqrt(var(hrerrors))))
println(string("75th percentile: ", sort(hrerrors)[Integer(round(.75 * length(hrerrors)))]))
