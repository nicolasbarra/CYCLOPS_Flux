module CYCLOPS_CircularStatsModule

using MultivariateStats, StatsBase

export Circular_Mean, Fischer_Circular_CorrelationMeasures, Jammalamadka_Circular_CorrelationMeasures, Circular_Error, Circular_Error_List

function Circular_Mean(phases::Array{Number, 1})
  sinterm = sum(sin.(phases))
  costerm = sum(cos.(phases))

  atan2(sinterm, costerm) #TODO: fix this
end

function Fischer_Circular_Correlations(rphases, sphases)
	n1 = length(rphases)
	n2 = length(sphases)

	num = n1

	rphases=mod.(rphases + 2*pi, 2*pi)
	sphases=mod.(sphases + 2*pi, 2*pi)

	numtot = 0.
	d1tot = 0.
	d2tot = 0.

	for i in 1:num, j in (i+1):num
			numeratorterm = sin(sphases[i] - sphases[j]) * sin(rphases[i] - rphases[j])
			denomterm1 = (sin(sphases[i] - sphases[j]))^2
			denomterm2 = (sin(rphases[i] - rphases[j]))^2
			numtot = numtot + numeratorterm
			d1tot = d1tot + denomterm1
			d2tot = d2tot + denomterm2
	end
	fischercor = numtot / (sqrt(d1tot) * sqrt(d2tot))

	fischercor
end

function Jammalamadka_Circular_Correlations(rphases, sphases)
	numtot = 0.
	d1tot = 0.
	d2tot = 0.

	bar = x -> mod.(2*pi + Circular_Mean(x), 2*pi)
	rbar = bar(rphases)
	sbar = bar(sphases)

	numtot = sum(sin.(rphases - rbar) .* sin.(sphases - sbar))
	dtot = x, y -> sqrt(sum(sin.(x - y) .^ 2))
	d1tot = dtot(rphases, rbar)
	d2tot = dtot(sphases, sbar)
	Jammalamadka = numtot / (d1tot * d2tot)

	Jammalamadka
end

#= This is a modification of he Jammalamadka Circular Correlation described in Topics in Circular Statistics. It is required beacuse the circular average is not well defined with circular uniform data. This measure should only be used when 1 or both of the data sets being compared are uniform =#
function Jammalamadka_Uniform_Circular_Correlations(rphases, sphases)
	rphases = mod.(rphases, 2*pi)
	sphases = mod.(sphases, 2*pi)

	r_minus_s_bar = mod(atan2(sum(sin.(rphases - sphases)), sum(cos.(rphases - sphases))), 2*pi)
	r_plus_s_bar = mod(atan2(sum(sin.(rphases + sphases)), sum(cos.(rphases + sphases))), 2*pi)

	bars = FindComponentAngles(r_plus_s_bar, r_minus_s_bar)
	rbar = bars[1]
	sbar = bars[2]

	numtot = sum(sin.(rphases - rbar) .* sin.(sphases - sbar))
	dtot = x, y -> sqrt(sum(sin.(x - y) .^ 2))
	d1tot = dtot(rphases, rbar)
	d2tot = dtot(sphases, sbar)

	Jammalamadka = numtot / (d1tot * d2tot)

	Jammalamadka
end

function Circular_Rank_Phases(rphases)
	num = length(rphases)
	rphases = mod.(rphases+2*pi,2*pi)
	rranks = tiedrank(rphases) #TODO: fix this
	rrankphases = rranks*2*pi / num

	rrankphases
end

function Jammalamadka_Rank_Circular_Correlations(rphases, sphases)

	rphases = Circular_Rank_Phases(rphases)
	sphases = Circular_Rank_Phases(sphases)

	r_minus_s_bar = mod(atan2(sum(sin.(rphases - sphases)), sum(cos.(rphases - sphases))), 2*pi)
	r_plus_s_bar = mod(atan2(sum(sin.(rphases+sphases)), sum(cos.(rphases+sphases))), 2*pi)

	Ntot = length(rphases)

	term1 = cos.(rphases - sphases - r_minus_s_bar)
	term2 = cos.(rphases + sphases - r_plus_s_bar)

	Jammalamadka = 1 / Ntot*(sum(term1)) - 1 / Ntot * (sum(term2))

	Jammalamadka
end

function FindComponentAngles(angle_sum, angle_diff)
  rang = (angle_sum + angle_diff) / 2
  sang = (angle_sum - angle_diff) / 2

	[rang,sang]
end

function Fischer_Circular_CorrelationMeasures(rphases, sphases)
	rrankphases = Circular_Rank_Phases(rphases)
	srankphases = Circular_Rank_Phases(sphases)

	F = Fischer_Circular_Correlations(rphases, sphases)
	FR = Fischer_Circular_Correlations(rrankphases, srankphases)

	[F, FR]
end

function Jammalamadka_Circular_CorrelationMeasures(rphases, sphases)
	J = Jammalamadka_Circular_Correlations(rphases, sphases)
	JU = Jammalamadka_Uniform_Circular_Correlations(rphases, sphases)
	JR = Jammalamadka_Rank_Circular_Correlations(rphases, sphases)

	[J, JU, JR]
end

function Circular_Error(truth, estimate)
	truth = mod(2*pi + truth, 2*pi)
	estimate = mod(2*pi + estimate, 2*pi)

	diff1 = mod((truth - estimate), 2*pi)
	diff2 = diff1 - (2*pi)

	min(diff1, diff2)
end

function Circular_Error_List(true_list, estimate_list)
	n1 = length(true_list)
	n2 = length(estimate_list)

	if (n1 != n2)
		print("Error Warning")
	end

	error_list = zeros(n1)

	for count in 1:n1
		error_list[count] = Circular_Error(true_list[count], estimate_list[count])
	end

	error_list
end

end # CYCLOPS_CircularStatsModule
