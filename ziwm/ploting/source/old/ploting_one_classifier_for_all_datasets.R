library(ggplot2)

script.dir <- dirname(sys.frame(1)$ofile)

dataset_name <- "svm_bagging_results"
#dataset_name <- "elm_final_results"

csv_name <- paste(dataset_name, ".csv", sep="")
csv_path <- file.path(script.dir, "..", "csv", csv_name)
data <- read.table(csv_name, sep =",", header = TRUE)

ensembles = data
ensembles = subset(ensembles, dataset != "fertility_yesnoreal")
ensembles = subset(ensembles, dataset != "fertility_yesnoOHE")
#ensembles = subset(ensembles, model != "MockClassifier")

ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")

ggplot(ensembles, aes(ensemble_size, score, colour=ensemble_all)) + 
  geom_point() + geom_line() + facet_wrap(~dataset) # + geom_smooth(aes(group=ensemble_all), se = FALSE)

png_name <- paste(dataset_name, ".png", sep="")
png_path <- file.path(script.dir, "..", "png", png_name)
ggsave(png_path, width = 40, height = 20, units = "cm")

