module CYCLOPS_SeedModule

using Statistics: mean, std

export cleandata!, getseed, getseed_mca, getseed_homologuesymbol, getseed_homologueprobe, dispersion!, dispersion, getseed_homologuesymbol_brain

function cleandata!(data::Array{Float64, 2}, bluntpercent)
	ngenes, nsamples = size(data)
	nfloor = Int(1 + floor((1 - bluntpercent) * nsamples))
	nceiling = Int(ceil(bluntpercent*nsamples))
	for row in 1:ngenes
		sorted = sort(vec(data[row, :]))
		vfloor = sorted[nfloor]
		vceil = sorted[nceiling]
		for sample in 1:nsamples
			data[row, sample] = max(vfloor, data[row, sample])
			data[row,sample] = min(vceil, data[row, sample])
		end
	end

	data
end

function getseed(data::Array{Any, 2}, symbol_list, maxcv, mincv, minmean, blunt)
	data_symbols = data[2:end, 2]
	data_data = data[2:end, 4:end]
	data_data = Array{Float64}(data_data)
	data_data = cleandata!(data_data, blunt)
	ngenes, namples = size(data_data)

	gene_means = mean(data_data,2)
	gene_sds = std(data_data,2)
	gene_cvs = gene_sds ./ gene_means

	criteria1 = findall(data_symbols, symbol_list)
	criteria2 = findall((gene_means .> minmean), true)
	criteria3 = findall((gene_cvs .> mincv), true)
	criteria4 = findall((gene_cvs .< maxcv), true)

	allcriteria = intersect(criteria1, criteria2, criteria3, criteria4)
	seed_data = data_data[allcriteria,:]
	seed_symbols = data_symbols[allcriteria, :]

	seed_symbols, seed_data
end

function getseed_mca(data::Array{Any, 2}, probe_list, maxcv, mincv, minmean, blunt)
	data_symbols = data[2:end, 2]
	data_probes = data[2:end, 1]

	data_data = data[2:end, 4:end]
	data_data = Array{Float64}(data_data)
	data_data = cleandata!(data_data, blunt)
	ngenes, namples = size(data_data)

	gene_means = mean(data_data, 2)
	gene_sds = std(data_data, 2)
	gene_cvs = gene_sds ./ gene_means

	criteria1 = findall(data_probes, probe_list)
	criteria2 = findall((gene_means .> minmean), true)
	criteria3 = findall((gene_cvs .> mincv), true)
	criteria4 = findall((gene_cvs .< maxcv), true)

	allcriteria = intersect(criteria1, criteria2, criteria3, criteria4)
	seed_data = data_data[allcriteria, :]
	seed_symbols = data_symbols[allcriteria, :]

	seed_symbols, seed_data
end

function getseed_homologuesymbol(data::Array{Any, 2}, symbol_list, maxcv, mincv, minmean, blunt)
	data_symbols = data[2:end, 2]
	data_probes = data[2:end, 1]

	data_data = data[2:end, 3:end]
	data_data = Array{Float64}(data_data)
	data_data = cleandata!(data_data, blunt)
	ngenes, namples = size(data_data)

	gene_means = mean(data_data, 2)
	gene_sds = std(data_data,2)
	gene_cvs = gene_sds ./ gene_means

	criteria1 = findall(data_symbols, symbol_list)
	criteria2 = findall((gene_means .> minmean), true)
	criteria3 = findall((gene_cvs .> mincv), true)
	criteria4 = findall((gene_cvs .< maxcv), true)

	allcriteria = intersect(criteria1, criteria2, criteria3, criteria4)
	seed_data = data_data[allcriteria, :]
	seed_symbols = data_symbols[allcriteria, :]

	seed_symbols, seed_data
end

function getseed_homologueprobe(data::Array{Any, 2}, probe_list, maxcv, mincv, minmean, blunt)
	data_symbols = data[2:end, 2]
	data_probes = data[2:end, 1]

	data_data = data[2:end, 3:end]
	data_data = Array{Float64}(data_data)
	data_data = cleandata!(data_data, blunt)
	ngenes,namples = size(data_data)

	gene_means = mean(data_data, 2)
	gene_sds = std(data_data, 2)
	gene_cvs = gene_sds ./ gene_means

	criteria1 = findall(data_probes, probe_list)
	criteria2 = findall((gene_means .> minmean), true)
	criteria3 = findall((gene_cvs .> mincv), true)
	criteria4 = findall((gene_cvs .< maxcv), true)

	allcriteria = intersect(criteria1, criteria2, criteria3, criteria4)
	seed_data = data_data[allcriteria, :]
	seed_symbols = data_symbols[allcriteria, :]

	seed_symbols, seed_data
end

function dispersion!(data::Array{Float64, 2})
	ngenes, nsamples = size(data)
	for gene in 1:ngenes
		genemean = mean(data[gene, :])
		for sample = 1:nsamples
			data[gene, sample] = (data[gene, sample] - genemean) / genemean
		end
	end

	data
end

function dispersion(data::Array{Float64, 2})
	ngenes, nsamples = size(data)
	ndata = zeros(ngenes, nsamples)
	for gene in 1:ngenes
		genemean = mean(data[gene, :])
		for sample in 1:nsamples
			ndata[gene, sample] = (data[gene, sample] - genemean) / genemean
		end
	end

	ndata
end

function getseed_homologuesymbol_brain(data_data, symbol_list, data_symbols, maxcv, mincv, minmean, blunt)
	data_data = Array{Float64}(data_data)
	data_data = cleandata!(data_data, blunt)
	ngenes, namples = size(data_data)

	gene_means = vec(mean(data_data, dims=2))
	gene_sds = vec(std(data_data; dims=2))
	gene_cvs = gene_sds ./ gene_means

	criteria1 = findall(in(symbol_list), data_symbols)
	criteria2 = findall(gene_means .> minmean)
	criteria3 = findall(gene_cvs .> mincv)
	criteria4 = findall(gene_cvs .< maxcv)

	allcriteria = intersect(criteria1, criteria2, criteria3, criteria4)
	seed_data = data_data[allcriteria, :]
	seed_symbols = data_symbols[allcriteria, :]

	seed_symbols, seed_data
end

end  # module CYCLOPS_SeedModule
