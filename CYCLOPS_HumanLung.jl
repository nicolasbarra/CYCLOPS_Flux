using Distributed, CSV

@everywhere basedir = homedir()
@everywhere cd(basedir * "/Downloads/Research")
fullnonseed_data_laval = read("AnnotatedLAVALData_March3_2015.csv")
fullnonseed_data_grng = read("AnnotatedGRNGData_March3_2015.csv")
seed_homologues = read("LungCyclerHomologues.csv")
#homologue_symbol_list = seed_homologues[2:end,2]
