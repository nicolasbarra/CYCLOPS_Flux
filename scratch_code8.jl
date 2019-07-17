using Flux
A1 = Dense(5, 5)
B1 = Dense(5, 5)
A2 = Dense(5, 5)
B2 = Dense(5, 5)
C2 = Dense(5, 5)
D = Dense(5, 5)
E = Dense(5, 5)
model = Chain(x -> cat(Chain(A1, B1)(x), Chain(A2, B2, C2)(x); dims=3), D, E)
model(rand(5))

ERROR: MethodError: no method matching *(::TrackedArray{…,Array{Float32,2}}, ::TrackedArray{…,Array{Float32,3}})
