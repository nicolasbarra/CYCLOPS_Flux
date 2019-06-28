module CYCLOPS_FluxAutoEncoderModule

export makeautoencoder

function makeautoencoder(in_out_dim::Integer, n_circs::Integer, n_lins::Integer)
    if n_circs == 0 && n_lins == 0
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot both be zero."))
    elseif n_circs < 0 || n_lins < 0
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot be less than zero."))
    end
    encoder = Dense(in_out_dim, 2)
    function circ(x)
      length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
      x./sqrt(sum(x .* x))
    end
    lin = Dense(2, 2)
    function bottleneck(n_circs1, n_lins1)
        if n_circs1 == 0
            x -> reduce(vcat, Iterators.repeated(lin(x), n_lins1))
        elseif n_lins1 == 0
            x -> reduce(vcat, Iterators.repeated(circ(x), n_circs1))
        else
            x -> vcat(reduce(vcat, Iterators.repeated(circ(x), n_circs1)), reduce(vcat, Iterators.repeated(lin(x), n_lins1)))
        end
    end
    decoder = Dense(n_circs*2 + n_lins*2, in_out_dim)
    model = Chain(encoder, bottleneck(n_circs, n_lins), decoder)

    model
end

end  # module CYCLOPS_FluxAutoEncoderModule
