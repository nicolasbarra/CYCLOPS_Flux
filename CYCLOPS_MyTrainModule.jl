module CYCLOPS_MyTrainModule

import Statistics: mean
import Flux.Tracker: Params, gradient, update!
import Juno: @progress

export mytrain!

call(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call, fs)
struct StopException <: Exception end

function mytrain!(loss, ps, data, opt; cb = () -> ())
  lossrec =[]
  ps = Params(ps)
  cb = runall(cb)
  @progress for d in data
    try
      gs = gradient(ps) do
        loss(d...)
      end
      update!(opt, ps, gs)
      append!(lossrec, loss(d...))
      cb()
    catch ex
      if ex isa StopException
        break
      else
        rethrow(ex)
      end
    end
  end
  avg = mean(lossrec)
  println(string("Average loss this epoch: ", avg))

  avg
end

end  # module CYCLOPS_MyTrainModule
