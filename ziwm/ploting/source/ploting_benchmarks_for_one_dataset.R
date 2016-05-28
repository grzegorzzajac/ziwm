library(ggplot2)

dataset_name <- "tic_tac_toe_OHE"

csv_name <- paste(dataset_name, ".csv", sep="")
csv_path <- file.path("..", "csv", csv_name)
data <- read.table(csv_path, sep =",", header = TRUE)

ensembles = data
#ensembles = subset(ensembles, dataset == "house_votes_84")
ensembles = subset(ensembles, model != "MockClassifier")

ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")

ggplot(ensembles, aes(ensemble_size, score, colour=ensemble_all)) + 
  geom_point() + geom_line() + facet_wrap(~ensemble_all) # + geom_smooth(aes(group=ensemble_all), se = FALSE)

png_name <- paste(dataset_name, ".png", sep="")
png_path <- file.path("..", "png", png_name)
ggsave(png_path, width = 40, height = 20, units = "cm")

