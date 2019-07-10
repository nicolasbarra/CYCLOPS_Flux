function expr(op, inp...)
           expr = Expr(:call, op, inp)
           return expr
end

ex = expr(:vcat, ([1], [2]))
:(1 + 4 * 5)

julia> eval(ex)
21


macro vcatmac(n_circs1)
      encoderbottle = x -> circ(Dense(in_out_dim1, 2)(x)
      for i in 1:n_circs1
            vcat
end

function makecircs(funcs)
      if length(funcs) == 1
            pop!(funcs)
      elseif length(funcs) != 0
            vcat(pop!(funcs), makecircs(funcs))
      end
end

ints = []
for i in 1:8
      append!(ints, @eval $(Symbol("x_$i")) = Dense(in_out_dim1, 2))
end

for n in 1:10
      @eval $(Symbol("encoderbottle_$n")) = x -> circ(Dense(7, 2)(x))
end

@eval function makecircs(in_out_dim1, n_circs1)
      $([:($(Symbol("encoderbottle_$i")) = x -> circ(Dense(in_out_dim1, 2)(x))) for i in 1:n_circs1]...)
      ($([Symbol("encoderbottle_$i") for i in 1:n_circs1]...),)
end

@eval function f()
           $([:($(Symbol("x_$i")) = Dense(7, 2)) for i in 1:10]...)
       end
