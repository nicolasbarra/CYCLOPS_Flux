module CYCLOPS_PrePostProcessModule

using DataFrames
using Distributed: @spawn
using Statistics: mean
using LinearAlgebra: svd
using StatsBase: fit, zscore
using MultivariateStats: outdim, transform, llsq
using Distributions: sample, var, FDist, cdf

export makefloat!, getEigengenes, get_N_Eigengenes, PCA_transform_seed_data, repeat_PCA_transform_data, row_shuffle, Bonferroni_adjust, best_shift_cos, best_shift_cos2, cosinor_stats, multicore_cosinor_stats, compile_multicore_cosinor_stats

#= make all the columns of a DataFrame of type Float64, not String, since they are Numbers
function makefloat!(df)
    for col in 1:size(df)[2]
        if typeof(df[:, col]) == Array{String,1} # used to be a an Array{String,1}, now is a WeakRefStrings.StringArray{String,1}. However, a single index is still a String
            df[:, col] = map(x -> tryparse(Float64, x), df[:, col])
        end
    end
end
=#

#= New makefloat! function =#
function makefloat!(ar::Array{Any}) # convert to Array{Any} first, using convert{Matrix, df}.
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float32, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float32, 2}, ar)
end

function makefloat!(ar::Array{Any,1}, dim::Integer)
    for col in 1:size(ar)[1]
        if typeof(ar[col]) == String
            ar[i] = parse(Float32, ar[i])
        end
        if dim == 2
            convert(Array{Float32,2}, ar')
        end
    end
end

function makefloat!(df::DataFrame) # will convert to Array{Float} first
    ar = convert(Matrix, df)
    for col in 1:size(ar)[2]
        for row in 1:size(ar)[1]
            if typeof(ar[row,col]) == String
                ar[row, col] = parse(Float32, ar[row,col])
            end
        end
    end
    ar = convert(Array{Float32, 2}, ar)
end


function getEigengenes(numeric_data::Array{Float64, 2}, fraction_var::Number, dfrac_var::Number, maxeig::Number)
    svd_obj = svd(numeric_data)
    expvar = cumsum(svd_obj.S.^2, dims = 1) / sum(svd_obj.S.^2)
    ReductionDim1 = 1 + length(expvar[expvar .<= fraction_var])
    vardif = diff(expvar, dims = 1)
    ReductionDim2 = 1 + length(vardif[vardif .>= dfrac_var])
    ReductionDim = min(ReductionDim1, ReductionDim2, maxeig)
    Transform = svd_obj.V[:, 1:ReductionDim]'

    ReductionDim, Array{Float32,2}(10*Transform)
end

function get_N_Eigengenes(numeric_data::Array{Float64, 2}, nkeep::Integer)
    svd_obj = svd(numeric_data)
    Transform = v[:,1:nkeep]'

    10*Transform
end

function PCA_transform_seed_data(numeric_data::Array{Float64, 2}, fraction_var::Number, maxeig::Number)
    Mod = fit(PCA, numeric_data; pratio = fraction_var, maxoutdim = maxeig)
    ReductionDim = outdim(Mod)
    Transform = transform(Mod, numeric_data)
    Transform = 10*Transform / maximum(Transform)

    ReductionDim, Transform
end

function repeat_PCA_transform_data(numeric_data::Array{Float64, 2}, outsize::Integer)
    z_data = zscore(numeric_data, 2)
    Mod = fit(PCA, z_data; pratio = 1., maxoutdim = outsize)
    Transform = transform(Mod, z_data)

    Transform
end

# row_shuffle independantly permutes the elements in each row of a matrix
function row_shuffle(data_matrix::Array{Float64, 2})
    (nrows, ncols) = size(data_matrix)
    shuffled_matrix = zeros(nrows, ncols)
    for r in 1:nrows
        shuffled_matrix[r, :] = data_matrix[r, sample(1:ncols, ncols; replace = false)]
    end

    shuffled_matrix
end

function Bonferroni_adjust(pvalues::Array{Float64, 1})
    n = size(pvalues)[1]
    adjusted = zeros(n)
    for cnt in 1:n
        adjusted[cnt] = min(n * pvalues[cnt], 1.)
    end

    adjusted
end

function best_shift_cos(list1, list2, conversion_flag)
    nlist = length(list1)

    if conversion_flag == "hours"
        llist2 = mod.(list2, 24) * (pi / 12)
    elseif conversion_flag == "radians"
        llist2 = mod.(list2, 2*pi)
    else
        println("FLAG ERROR")
    end

    bestdist = 10
    bestlist = zeros(nlist)
    n = 0

    function bestlistfinder(f, x)
        for a in range(-pi, stop = pi, length = 192)
            n = n + 1
            llist1 = mod.(f(x .+ a), 2*pi)
            runningscore = 0
            for i in 1:nlist
                cosd = (1 - cos(llist1[i] - llist2[i])) / 2
                runningscore = runningscore + cosd
            end
            adist = runningscore / nlist
            if bestdist > adist
                bestdist = adist
                bestlist = llist1
            end
        end
    end

    bestlistfinder(+, list1)
    bestlistfinder(-, list1)

    bestlist
end

function best_shift_cos2(acrophaselist1, acrophaselist2, samplephase1, conversion_flag)
    if conversion_flag == "hours"
        llist2 = mod.(acrophaselist2, 24) * (pi / 12)
    elseif conversion_flag == "radians"
        llist2 = mod.(acrophaselist2, 2*pi)
    else
        println("FLAG ERROR")
    end

    bestdist = 10
    bestacrophaselist1 = acrophaselist1
    bestsamplephase1 = samplephase1

    function bestsamplephase1_finder(x, y)
        for a in range(-pi, stop = pi, length = 192)
            llist1 = mod.(x, 2*pi)
            adist = mean(1 - cos.(float(llist1 - llist2)))
            if bestdist > adist
                bestdist = adist
                bestacrophaselist1 = llist1
                bestsamplephase1 = mod.(y, 2*pi)
            end
        end
    end

    bestsamplephase1_finder(acrophaselist1 + a, samplephase1 + a)
    bestsamplephase1_finder(-(acrophaselist1 + a), -(samplephase1 + a))

    bestacrophaselist1, bestsamplephase1
end

function cosinor_stats(expression::Array{Float64, 1}, o_PREDICTED_PHASELIST::Array{Float64, 1}, n_shifts=20)
    len = size(o_PREDICTED_PHASELIST)[1]
    shift_list = range(2*pi / n_shifts, stop = 2*pi, length = n_shifts)

    min_error = var(expression) * 10E20
    best_shift = "error"
    for shift in shift_list
        l_PREDICTED_PHASELIST = mod.((o_PREDICTED_PHASELIST + shift), (2*pi))
        XLINEAR = reshape(l_PREDICTED_PHASELIST, len,1)
        mod_linear = llsq(XLINEAR,expression)
        m_linear, b_linear = mod_linear[1:end-1], mod_linear[end]
        predict_linear = XLINEAR * m_linear + b_linear
        sse_linear = sum(abs2.(expression - predict_linear))
        if sse_linear < min_error
            min_error = sse_linear
            best_shift = shift
        end
    end

    l_PREDICTED_PHASELIST = mod.((o_PREDICTED_PHASELIST + best_shift), (2*pi))
    SIN_l_PREDICTED_PHASELIST = sin.(l_PREDICTED_PHASELIST)
    COS_l_PREDICTED_PHASELIST = cos.(l_PREDICTED_PHASELIST)

    XLINEAR = reshape(l_PREDICTED_PHASELIST, len, 1)
    XFULL = hcat(l_PREDICTED_PHASELIST, SIN_l_PREDICTED_PHASELIST, COS_l_PREDICTED_PHASELIST)

    mod_linear = llsq(XLINEAR, expression)
    m_linear, b_linear = mod_linear[1:end-1], mod_linear[end]

    mod_full = llsq(XFULL,expression)
    m_full, b_full = mod_full[1:end-1], mod_full[end]

    predict_linear = XLINEAR * m_linear + b_linear
    predict_full = XFULL * m_full + b_full

    sse_linear = sum(abs2.(expression - predict_linear))
    sse_full = sum(abs2.(expression - predict_full))

    f_metric = ((sse_linear - sse_full) / 2) / ((sse_full) /(len - 5)) # p2=4+1 (+1 =shift) ,p1=2+1  (+1 =shift) ,n=len
    pval = 1 - cdf(FDist(2, len - 5), f_metric) # Note - null hypothesis this satisifes F distribution, with (p2−p1, n−p2) degrees of freedom.


    #####################
    #Get Phase, Amp, Avg#
    #####################
    SIN_o_PREDICTED_PHASELIST = sin.(o_PREDICTED_PHASELIST)
    COS_o_PREDICTED_PHASELIST = cos.(o_PREDICTED_PHASELIST)
    XCOSINOR = hcat(SIN_o_PREDICTED_PHASELIST, COS_o_PREDICTED_PHASELIST)
    mod_cosinor = llsq(XCOSINOR, expression)

    m_cosinor, b_cosinor = mod_cosinor[1:end-1], mod_cosinor[end]

    predict_cosinor = XCOSINOR * m_cosinor + b_cosinor

    sse_cosinor = sum(abs2.(expression - predict_cosinor))
    mean_expression = mean(expression)
    sse_base = sum(abs2.(expression - mean_expression))

    r2 = 1 - (sse_cosinor / sse_base)

    sinterm, costerm = mod_cosinor[1], mod_cosinor[2]
    acrophase = atan(sinterm, costerm)
    amp = sqrt(sinterm^2 + costerm^2)
    fitavg = b_cosinor[1]
    avg = mean(expression)

    pval, acrophase, amp, fitavg, avg, r2
end

function multicore_cosinor_stats(data::Array{Float64, 2}, PREDICTED_PHASELIST::Array{Float64, 1}, n_shifts=20)
    ngenes, nsamples=size(data)
    PrbPval = zeros(ngenes)
    PrbPhase = zeros(ngenes)
    PrbAmp = zeros(ngenes)
    PrbFitMean = zeros(ngenes)
    PrbMean = zeros(ngenes)
    PrbR2 = zeros(ngenes)

    for row in 1:ngenes
        PrbPval[row], PrbPhase[row], PrbAmp[row], PrbFitMean[row], PrbMean[row], PrbR2[row]  = Corrected_Cosinor_Statistics_Faster(data[row, :], PREDICTED_PHASELIST, n_shifts)
    end

    PrbPval, PrbPhase, PrbAmp, PrbFitMean, PrbMean, PrbR2
end

function compile_multicore_cosinor_stats(annotated_data::Array{Any, 2}, PREDICTED_PHASELIST::Array{Float64, 1}, NumericStartCol::Number, n_shifts::Number)
    alldata_annot = annotated_data[1:end, 1:(NumericStartCol - 1)]
    alldata_data = Array{Float64}(annotated_data[2:end, NumericStartCol:end])

    ngenes = size(alldata_data)[1]
    rowbin = Int(floor(ngenes / 5))
    @time begin
        estimated_phaselist = vec(PREDICTED_PHASELIST)
        c1 = @spawn MultiCore_Cosinor_Statistics(alldata_data[(1:(1*rowbin)), :], estimated_phaselist)
        c2 = @spawn MultiCore_Cosinor_Statistics(alldata_data[((1 + 1 * rowbin):(2 * rowbin)), :], estimated_phaselist, n_shifts)
        c3 = @spawn MultiCore_Cosinor_Statistics(alldata_data[((1 + 2 * rowbin):(3 * rowbin)), :], estimated_phaselist, n_shifts)
        c4 = @spawn MultiCore_Cosinor_Statistics(alldata_data[((1 + 3 * rowbin):(4 * rowbin)), :], estimated_phaselist, n_shifts)
        c5 = @spawn MultiCore_Cosinor_Statistics(alldata_data[((1 + 4 * rowbin):ngenes), :], estimated_phaselist, n_shifts)

        PrbPval1, PrbPhase1, PrbAmp1, PrbFitMean1, PrbMean1, PrbRsq1 = fetch(c1)
        PrbPval2, PrbPhase2, PrbAmp2, PrbFitMean2, PrbMean2, PrbRsq2 = fetch(c2)
        PrbPval3, PrbPhase3, PrbAmp3, PrbFitMean3, PrbMean3, PrbRsq3 = fetch(c3)
        PrbPval4, PrbPhase4, PrbAmp4, PrbFitMean4, PrbMean4, PrbRsq4 = fetch(c4)
        PrbPval5, PrbPhase5, PrbAmp5, PrbFitMean5, PrbMean5, PrbRsq5 = fetch(c5)
    end

    PrbPval = vcat(PrbPval1, PrbPval2, PrbPval3, PrbPval4, PrbPval5)
    PrbPhase = vcat(PrbPhase1, PrbPhase2, PrbPhase3, PrbPhase4, PrbPhase5)
    PrbAmp = vcat(PrbAmp1, PrbAmp2, PrbAmp3, PrbAmp4, PrbAmp5)
    PrbMean = vcat(PrbMean1, PrbMean2, PrbMean3, PrbMean4, PrbMean5)
    PrbFitMean = vcat(PrbFitMean1, PrbFitMean2, PrbFitMean3, PrbFitMean4, PrbFitMean5)
    PrbRsq=vcat(PrbRsq1,PrbRsq2,PrbRsq3,PrbRsq4,PrbRsq5)

    PrbPtr = (PrbAmp + PrbFitMean) ./  (PrbFitMean - PrbAmp)
    PrbBon = Bonferroni_Adjust(PrbPval)

    cosinor_output = hcat(PrbPval, PrbBon, PrbPhase, PrbAmp, PrbFitMean, PrbMean, PrbRsq, PrbPtr)
    headers = ["pval", "bon_pval", "phase", "amp", "fitmean", "mean", "rsq", "ptr"]
    headers = reshape(headers, 1, length(headers))

    cosinor_output = vcat(headers, cosinor_output)
    cosinor_output = hcat(alldata_annot, cosinor_output)

    cosinor_output
end

end  # module CYCLOPS_PrePostProcessModule
