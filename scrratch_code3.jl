using Flux
using BSON: @save

import Random
Random.seed!(12345)


include("CYCLOPS_TrainingModule.jl")

encoderA = Dense(10, 2)
circ(x) = x./sqrt(sum(x .* x))
lin = Dense(2, 2)
decoderB = Dense(4, 10)
function makebottle(n)
    x -> reduce(vcat, Iterators.repeated(circ(x), n))
end

m1 = Chain(encoderA, makebottle(2), decoderB)

testA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
testB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test = cat(testA, testB; dims = 2)
test1 = mapslices(x -> [x], test, dims=1)[:]

loss(x) = Flux.mse(m1(x), x)

a = CYCLOPS_TrainingModule.mytrain!(loss, Flux.params(m1), zip(test1), ADAM())
#=
function extractphase(data_matrix, model)
    points = size(data_mXatrix, 2)
    phases = zeros(points)
    for n in 1:points
        phases[n] = model[1:2](data_matrix[:, n])
    end
    phases
end

estimated_phaselist = extractphase(test, m1)
estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
=#
#@save "m1.bson" m1
