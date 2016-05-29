# ploting
# score / ensemble size
# each dataset separately
# (uses all *.csv files in 'results' dir)

script.dir <- dirname(sys.frame(1)$ofile)
csv_path <- file.path(script.dir, "..", "..", "results")

file_names <- dir(csv_path)
print(file_names)

setwd(csv_path)
ensembles <- do.call(rbind,lapply(file_names, read.csv))

ensembles = subset(ensembles, dataset != "fertility_yesnoreal")
ensembles = subset(ensembles, dataset != "fertility_yesnoOHE")
#ensembles = subset(ensembles, model != "MockClassifier")

ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")

ggplot(ensembles, aes(ensemble_size, score, colour=ensemble_all)) + 
  geom_point() + geom_line() + facet_wrap(~dataset) # + geom_smooth(aes(group=ensemble_all), se = FALSE)

png_name <- paste("type1_score_size_dataset", ".png", sep="")
png_path <- file.path(script.dir, "..", "png", png_name)
ggsave(png_path, width = 120, height = 60, units = "cm")