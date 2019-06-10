using CSV

fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
alldata_times = fullnonseed_data[3, 4:end]

#println(mean(alldata_data,2))

alldata_data = fullnonseed_data[4:end, 4:end]





for col in 1:size(alldata_data)[2]
    if typeof(alldata_data[:, col]) != Array{Float64,1}
        println("foooyy")
    end
end

#show(alldata_data, allcols=true)



#println(first(alldata_data, 2))
#alldata_data = convert(Matrix, alldata_data)
