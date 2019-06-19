using Flux, Juno

encoderA = Dense(10, 2)
encoderB(x) =  atan(Tracker.data(x[2]), Tracker.data(x[1]))
decoderA(x) = [cos(x), sin(x)]
decoderB = Dense(2, 10)


m1 = Chain(encoderA, encoderB, decoderA, decoderB)

testA = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
testB = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
test = cat(testA, testB; dims=2)
test1 = mapslices(x -> [x], test, dims=1)[:]


loss(x) = Flux.mse(m1(x), x)

Flux.@epochs 400 Flux.train!(loss, Flux.params(m1), zip(test1), ADAM(), cb = () -> println(string("Loss: ", loss(test))))

function extractphase(data_matrix, model)
    points = size(data_matrix, 2)
    phases = zeros(points)
    for n in 1:points
        phases[n] = model[1:2](data_matrix[:, n])
    end
    phases
end

estimated_phaselist = extractphase(test, m1)
estimated_phaselist = mod.(estimated_phaselist .+ 2*pi, 2*pi)
