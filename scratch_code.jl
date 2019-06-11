using CSV

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")

function makefloat!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) != Array{Float64,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end

makefloat!(4, fullnonseed_data)

fullnonseed_data_array = convert(Matrix, fullnonseed_data)

println(fullnonseed_data_array)


#alldata_times = fullnonseed_data[3, 4:end]

#println(mean(alldata_data,2))

#alldata_data = fullnonseed_data[4:end, 4:end]





#= for col in 1:size(alldata_data)[2]
    if typeof(alldata_data[:, col]) != Array{Float64,1}
        println("foooyy")
    end
end =#

#show(alldata_data, allcols=true)



#println(first(alldata_data, 2))
#alldata_data = convert(Matrix, alldata_data)
