function circ(x)
  length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
  x./sqrt(sum(x .* x))
end

makecircs(in_out_dim1, n_circs1) = y -> begin
circlayers = []
for i in 1:n_circs1
      @eval $(Symbol("encoderbottle_$i")) = Chain(Dense(in_out_dim1, 2), circ)
end
for i in 1:n_circs1
  :(encoderbottle_$i)

for i in 1:n_circs1
  append!(circlayers, (x -> circ(Dense(in_out_dim1, 2)(x))))
end
circtups = tuple(circlayers...)
function arrayify(x)
  [x]
end
circtupsarr = arrayify.(circtups)
vcat(circtupsarr()...)
end




circlayerstup = vcat(enumerate(circlayers)...);
finalcirclayers = []
for i in circlayerstup
  append!(finalcirclayers, i[2])
end


for i in n_circs
vcat(circ(Dense(in_out_dim1, 2)(x)), 2)

makecircs(in_out_dim1) = x -> vcat([circ(Dense(in_out_dim1, 2)(x)), circ(Dense(in_out_dim1, 2)(x))])


makecircs(in_out_dim1, n_circs1) = y -> vcat( collect(Iterators.repeated(circ(y), n_circs1)))

makelins(in_out_dim1, lin_dim1, n_lins1) = x -> vcat( collect(Iterators.repeated(lin(Dense(in_out_dim1, lin_dim1)(x)), n_lins1)))

foo = x -> circ(a(x))
endcode = vcat(foo, foo)
endcode(g)
