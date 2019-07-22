module CYCLOPS_FluxAutoEncoderModule

#= This module provides a function that creates a balanced autoencoder using Flux. Specifically, this function creates a circular node autoencoder where it is possible to specificy the number of circular nodes and linear nodes in the bottleneck layer in addition to the input (and thus also output since this is an balanced autoencoder) dimensions. It then returns a model that reflects these inputs that has been created using Flux's Chain function. The bottleneck layer is the concatenation of the specified number of circular and/or linear layers. =#

import Flux: Dense, Chain
import Flux.Tracker: data, param

export makeautoencoder_naive
# TODO: Change exports when makeautoencoder is fixed.
#=
function makeautoencoder(in_out_dim::Integer, n_circs::Integer, n_lins::Integer, lin_dim::Integer)
    if n_circs == 0 && n_lins == 0
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot both be zero."))
    elseif n_circs < 0 || n_lins < 0
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot be less than zero."))
    elseif lin_dim < 1
        throw(ArgumentError("The input/output dimensions of the linear node(s) in the bottleneck layer must be at least 1."))
    end
    function circ(x)
      length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
      x./sqrt(sum(x .* x))
    end
    lin = Dense(lin_dim, lin_dim, x -> x)
    makelins(in_out_dim1, lin_dim1, n_lins1) = x -> vcat( collect(Iterators.repeated(lin(Dense(in_out_dim1, lin_dim1)(x)), n_lins1)))
    if n_circs == 0
        encodetobottleneck = makelins(in_out_dim, lin_dim, n_lins)
    elseif n_lins == 0
        encodetobottleneckprep = makecircs(in_out_dim, n_circs)
        encodetobottleneck = x -> encodetobottleneckprep(Dense(in_out_dim1, 2)(x))
    else
        encodetobottleneck = vcat(makecircs(in_out_dim, n_circs), makelins(in_out_dim, lin_dim, n_lins))
    end
    decoder = Dense(n_circs*2 + n_lins, in_out_dim)
    model = Chain(encodetobottleneck(x)[1], decoder)

    model
end
=#
#=
function makeautoencoder_naive(in_out_dim::Integer, n_circs::Integer, lin::Bool, lin_dim::Integer)
    if n_circs == 0 && lin == false
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot both be zero."))
    elseif n_circs < 0
        throw(ArgumentError("The number of circular nodes in the bottleneck cannot be less than zero."))
    elseif lin == true && lin_dim < 1
        throw(ArgumentError("The input/output dimensions of the linear node(s) in the bottleneck layer must be at least 1."))
    end
    function circ(x)
      length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
      x./sqrt(sum(x .* x))
    end
    if n_circs == 0
        println("linear layer autoencoder being created")
        encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
        encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
        encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
        encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
        encodetobottleneck = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
    elseif lin == false
        if n_circs == 1
            println("single circular layer autoencoder being created")
            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck = Chain(Dense(in_out_dim, 2), circ)
        elseif n_circs == 2
            println("double circular layer autoencoder being created")
            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneck2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneck(x) = vcat(encodetobottleneck1(x), encodetobottleneck2(x))
        else
            println("triple circular layer autoencoder being created")
            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneck2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneck3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneck(x) = vcat(encodetobottleneck1(x), encodetobottleneck2(x), encodetobottleneck3(x))
        end
    else
        if n_circs == 1
            println("single circular layer with linear layer autoencoder being created")
            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck(x) = vcat(encodetobottleneckcirc(x), encodetobottlenecklin(x))
        elseif n_circs == 2
            println("double circular layer with linear layer autoencoder being created")
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)

            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck(x) = vcat(encodetobottleneckcirc1(x), encodetobottleneckcirc2(x), encodetobottlenecklin(x))
        else
            println("triple circular layer with linear layer autoencoder being created")
            encodetobottleneckcirc1 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc2 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottleneckcirc3 = Chain(Dense(in_out_dim, 2), circ)
            encodetobottlenecklin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
            encodetobottleneck(x) = vcat(encodetobottleneckcirc1(x), encodetobottleneckcirc2(x), encodetobottleneckcirc3(x), encodetobottlenecklin(x))
        end
    end
    decoder = Dense(n_circs*2 + lin, in_out_dim)
    model = Chain(encodetobottleneck, decoder)

    model
end
=#

function makeautoencoder_naive(in_out_dim::Integer, n_circs::Integer, lin::Bool, lin_dim::Integer)
    if n_circs == 0 && lin == false
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot both be zero."))
    elseif n_circs < 0
        throw(ArgumentError("The number of circular nodes in the bottleneck cannot be less than zero."))
    elseif lin == true && lin_dim < 1
        throw(ArgumentError("The input/output dimensions of the linear node(s) in the bottleneck layer must be at least 1."))
    end
    function circ(x)
      length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
      x./sqrt(sum(x .* x))
    end
    if n_circs == 0
        encodetobottleneck = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, identity))
    elseif lin == false
        for i in 1:n_circs
            @eval $(Symbol("encoderbottle_$i")) = Chain(Dense($in_out_dim, 2), $circ)
        end
        modelmakerstring = "y -> vcat(u)"
        u = "encoderbottle_1(y)"
        if n_circs > 1
            for i in 2:n_circs
                u = u * ", encoderbottle_$i(y)"
            end
        end
        modelmakerstring = modelmakerstring[1:findfirst(isequal('u'), modelmakerstring) - 1] * u * modelmakerstring[findfirst(isequal('u'), modelmakerstring) + 1:end]

        encodetobottleneck = eval(Meta.parse(modelmakerstring))
    else
        for i in 1:n_circs
            @eval $(Symbol("encoderbottle_$i")) = Chain(Dense($in_out_dim, 2), $circ)
        end
        encoderbottle_lin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, identity))
        modelmakerstring = "y -> vcat(u, encoderbottle_lin(y))"
        u = "encoderbottle_1(y)"
        if n_circs > 1
            for i in 2:n_circs
                u = u * ", encoderbottle_$i(y)"
            end
        end
        modelmakerstring = modelmakerstring[1:findfirst(isequal('u'), modelmakerstring)-1] * u * modelmakerstring[findfirst(isequal('u'), modelmakerstring)+1:end]

        encodetobottleneck = eval(Meta.parse(modelmakerstring))
    end
    decoder = Dense(n_circs*2 + lin, in_out_dim)
    model = Chain(encodetobottleneck, decoder)

    model
end
#=
function makeautoencoder_string(in_out_dim::Integer, n_circs::Integer, lin::Bool, lin_dim::Integer)
    if n_circs == 0 && lin == false
        throw(ArgumentError("The number of circular nodes and linear layers in the bottleneck cannot both be zero."))
    elseif n_circs < 0
        throw(ArgumentError("The number of circular nodes in the bottleneck cannot be less than zero."))
    elseif lin == true && lin_dim < 1
        throw(ArgumentError("The input/output dimensions of the linear node(s) in the bottleneck layer must be at least 1."))
    end
    function circ(x)
      length(x) == 2 || throw(ArgumentError(string("Invalid length of input that should be 2 but is ", length(x))))
      x./sqrt(sum(x .* x))
    end
    if n_circs == 0
        encodetobottleneck = "Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x)"
    elseif lin == false
        for i in 1:n_circs
            @eval $(Symbol("encoderbottle_$i")) = Chain(Dense($in_out_dim, 2), $circ)
        end
        modelmakerstring = "y -> vcat(u)"
        u = "encoderbottle_1(y)"
        if n_circs > 1
            for i in 2:n_circs
                u = u * ", encoderbottle_$i(y)"
            end
        end
        modelmakerstring = modelmakerstring[1:findfirst(isequal('u'), modelmakerstring) - 1] * u * modelmakerstring[findfirst(isequal('u'), modelmakerstring) + 1:end]

        encodetobottleneck = eval(Meta.parse(modelmakerstring))
    else
        for i in 1:n_circs
            @eval $(Symbol("encoderbottle_$i")) = Chain(Dense($in_out_dim, 2), $circ)
        end
        encoderbottle_lin = Chain(Dense(in_out_dim, lin_dim), Dense(lin_dim, lin_dim, x -> x))
        modelmakerstring = "y -> vcat(u, encoderbottle_lin(y))"
        u = "encoderbottle_1(y)"
        if n_circs > 1
            for i in 2:n_circs
                u = u * ", encoderbottle_$i(y)"
            end
        end
        modelmakerstring = modelmakerstring[1:findfirst(isequal('u'), modelmakerstring)-1] * u * modelmakerstring[findfirst(isequal('u'), modelmakerstring)+1:end]

        encodetobottleneck = eval(Meta.parse(modelmakerstring))
    end
    decoder = Dense(n_circs*2 + lin, in_out_dim)
    model = Chain(encodetobottleneck, decoder)

    model
end
=#

# extracts the phase angles from the model for analysis
function extractphase(data_matrix, model, n_circs::Integer)
    points = size(data_matrix, 2)
    phases = zeros(n_circs, points)
    base = 0
    for circ in 1:n_circs
        for n in 1:points
            pos = model(data_matrix[:, n])
            phases[circ, n] = data(atan(pos[2 + base], pos[1 + base]))
        end
        base += 2
    end

    phases
end

end  # module CYCLOPS_FluxAutoEncoderModule
