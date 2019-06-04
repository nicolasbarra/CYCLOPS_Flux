#module CYCLOPS_FluxAutoEncoderModule_multi

W = randn(3, 5)
b = zeros(3)
x = rand(5)
gradient(myloss, W, b, x)

#end
