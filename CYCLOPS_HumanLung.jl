using Distributed, CSV, DataFrames

@everywhere basedir = homedir()
@everywhere cd(basedir * "/Downloads/Research")
fullnonseed_data_laval = CSV.read("AnnotatedLAVALData_March3_2015.csv")
fullnonseed_data_grng = CSV.read("AnnotatedGRNGData_March3_2015.csv")
homologue_symbol_list = CSV.read("LungCyclerHomologues.csv")[2]

#  add labels for one hot to each dataset
fullnonseed_data_laval.label = 1
fullnonseed_data_grng.label = 2

#  eliminate duplicate rows
deletecols!(fullnonseed_data_grng, [2, 3])

#  join the the DataFrames
fullnonseed_data = join(fullnonseed_data_laval, fullnonseed_data_grng, on = :Column1, makeunique = true)
