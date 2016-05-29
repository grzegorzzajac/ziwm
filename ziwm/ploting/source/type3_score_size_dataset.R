# ploting
# score / ensemble size
# each dataset has own image
# (uses all *.csv files in 'results' dir)

library(ggplot2)
library(tools) 

script.dir <- dirname(sys.frame(1)$ofile)
csv_path <- file.path(script.dir, "..", "..", "results")

file_names <- dir(csv_path)
print(file_names)

setwd(csv_path)
ensembles <- do.call(rbind,lapply(file_names, read.csv))

ensembles = subset(ensembles, dataset != "fertility_yesnoOHE")
ensembles = subset(ensembles, dataset != "fertility_yesnoreal")

ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")

unique_datasets = unique(ensembles$dataset)

for (unique_dataset in unique_datasets) {
  #print(unique_dataset)
  current_dataset = subset(ensembles, dataset == unique_dataset)
  
  ggplot(current_dataset, aes(ensemble_size, score, colour=ensemble_all)) + 
    geom_point() + geom_line()
  
  png_name <- paste("type3_score_size_dataset_", unique_dataset, ".png", sep="")
  png_path <- file.path(script.dir, "..", "png", png_name)
  ggsave(png_path, width = 40, height = 20, units = "cm")
}  