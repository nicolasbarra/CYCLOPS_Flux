module CYCLOPS_MyTrainSuppportModule

import Flux.Tracker: Params, gradient, data, update!

call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)
struct StopException <: Exception end

end  # module CYCLOPS_MyTrainSuppportModule
