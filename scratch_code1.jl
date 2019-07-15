using CSV, Statistics

function makefloat!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) == Array{String,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end

function makefloatfull!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) != Array{Float64,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end

MaxSeeds = 10000


fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
makefloatfull!(4, fullnonseed_data)


alldata_probes = fullnonseed_data[3:end, 1]



alldata_data = fullnonseed_data[4:end, 4:end]


makefloat!(1, alldata_data)


alldata_data = convert(Matrix, alldata_data)

n_probes = length(alldata_probes)

cutrank = n_probes - MaxSeeds
println(cutrank)

a = (sort(vec(mean(alldata_data, dims=2))))
println(a[cutrank])

for i in 1:cutrank
    println(a[i])
end
#=
Seed_MinMean = (sort(vec(mean(alldata_data, dims=2))))[cutrank]
println(Seed_MinMean)  # Note that this number is slightly different in new version versus old (42.889112582772285 here versus 42.88460199555892 in the old file) this is likely due to the fact that my method removes null values better (maybe?))
=#





#=function makefloat!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) != Array{Float64,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end =#

#makefloat!(4, fullnonseed_data)

#fullnonseed_data_array = convert(Matrix, fullnonseed_data)

#println(fullnonseed_data_array)


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
