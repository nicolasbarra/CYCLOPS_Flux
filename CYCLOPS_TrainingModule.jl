module CYCLOPS_TrainingModule

#= This module offers modificatied versions of Flux's train! function and @epochs macro to suit our needs better for our neural network, and the releveant supporting code. Specifically, we modify train! so that it stores the loss over the entire epoch (which is the same as the instance it is called) and returns the average loss over that epoch, as well as printing it to the console to keep track of training progress. We modify @epochs so that it is able to store the output of the modified train function in an array and then return that array. This is useful for plotting how the average loss changes across epochs of training. Additionally, it includes a feature that will halt the inputted number of epochs of training if there are more than 5 consecutive epochs during which the average loss over a particular epoch increases compared to the previous epoch. =#

import Statistics: mean
import Flux.Tracker: Params, gradient, data, update!
import Juno: @progress

export mytrain!, @myepochs

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

macro myepochs(n, ex)
  return :(lossrecord = [];
  @progress for i = 1:$(esc(n))
      @info "Epoch $i"
      avgloss = $(esc(ex))
      if length(lossrecord) > 6 && avgloss > lossrecord[length(lossrecord)] && lossrecord[length(lossrecord)] > lossrecord[length(lossrecord) - 1] && lossrecord[length(lossrecord) - 1] > lossrecord[length(lossrecord) - 2]
        break
      else
        append!(lossrecord, avgloss)
      end
    end;
    lossrecord = map(x -> data(x), lossrecord);

    lossrecord)
end

end  # module CYCLOPS_TrainingModule
