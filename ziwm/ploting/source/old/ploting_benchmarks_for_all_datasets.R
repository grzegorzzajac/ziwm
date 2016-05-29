library(ggplot2)

#data <- read.table("test_bigger.csv", sep =",", header = TRUE)
data <- read.table("test_aggr.csv", sep =",", header = TRUE)

ensembles = data
ensembles = subset(ensembles, dataset != "breast_cancer")
ensembles = subset(ensembles, dataset != "breast_cancer_OHE")
ensembles = subset(ensembles, dataset != "tic_tac_toe")
#ensembles = subset(data, ensemble != "NONE")
#ensembles = subset(ensembles, dataset == "wine")
#ensembles = subset(ensembles, model != "MockClassifier")

ensembles$ensemble_all = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$ensemble_all = paste(ensembles$ensemble_all, ensembles$voting_system, sep="_")

#ggplot(ensembles, aes(ensemble_size, score, colors)) + geom_point(aes(colour=ensemble_all)) +
ggplot(ensembles, aes(ensemble_size, score, colour=ensemble_all)) + 
  geom_point() + geom_smooth(aes(group=ensemble_all), se = FALSE) + facet_wrap(~dataset)

