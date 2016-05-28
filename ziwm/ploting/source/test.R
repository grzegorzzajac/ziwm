library(ggplot2)
library(tools) 

all_files = list.files("../csv", pattern="*.csv", full.names=TRUE)

for (i in 1:length(all_files)) {
  dataset_path <- all_files[i]
  data <- read.table(dataset_path, sep =",", header = TRUE)
  
  ensembles = data
  ensembles = subset(ensembles, model != "MockClassifier")
  
  ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
  ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")
  
  ggplot(ensembles, aes(ensemble_size, score, colour=ensemble_all)) + 
    geom_point() + geom_line() + facet_wrap(~ensemble_all) # + geom_smooth(aes(group=ensemble_all), se = FALSE)
  
  dataset_file_name <- basename(dataset_path)
  dataset_name <- file_path_sans_ext(dataset_file_name)
  png_name <- paste(dataset_name, ".png", sep="")
  png_path <- file.path("..", "png", png_name)
  ggsave(png_path, width = 40, height = 20, units = "cm")
}
