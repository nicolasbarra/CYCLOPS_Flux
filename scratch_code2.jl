using CSV
function makefloatfull!(x, df)
    for col in x:size(df)[2]
        if typeof(df[:, col]) != Array{Float64,1}
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end
fullnonseed_data = CSV.read("Annotated_Unlogged_BA11Data.csv")
makefloatfull!(4, fullnonseed_data)



alldata_data = fullnonseed_data[3:end, 4:end]

alldata_data = convert(Matrix, alldata_data)
