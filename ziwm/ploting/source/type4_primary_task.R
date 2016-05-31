# ploting
# only 5 models - primary task
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
ensembles = subset(ensembles, dataset != "glass")
ensembles = subset(ensembles, dataset != "soybean_large")

print(ensembles)

single_elm = subset(ensembles, model == "hpelmnn" & voting_system == "Arithmetic Mean" & ensemble == "Random Networks" & ensemble_size == 1)
single_bpnn = subset(ensembles, model == "Back Propagation PyBrain" & voting_system == "Arithmetic Mean" & ensemble == "Random Networks" & ensemble_size == 1)

print(single_elm)
print(single_bpnn)

#single_elm_score = single_elm$score
#print(single_elm_score)

ensembles = subset(ensembles, model != "svm")
ensembles = subset(ensembles, model != "Back Propagation PyBrain")
ensembles = subset(ensembles, voting_system != "Arithmetic Mean")

#ensembles = rbind(ensembles, single_elm, single_bpnn)

ensembles$MODELS = paste(ensembles$model, ensembles$ensemble, sep="_")
ensembles$MODELS = paste(ensembles$MODELS, ensembles$voting_system, sep="_")

unique_datasets = unique(ensembles$dataset)

for (unique_dataset in unique_datasets) {
  #print(unique_dataset)
  current_dataset = subset(ensembles, dataset == unique_dataset)
  
  single_elm_this_dataset = subset(single_elm, dataset == unique_dataset)
  single_elm_this_dataset_score = single_elm_this_dataset$score
  Single_ELM <- data.frame( x = c(-Inf, Inf), y = single_elm_this_dataset_score, Single_ELM = factor(single_elm_this_dataset_score) )
  
  single_bpnn_this_dataset = subset(single_bpnn, dataset == unique_dataset)
  single_bpnn_this_dataset_score = single_bpnn_this_dataset$score
  Single_BPNN <- data.frame( x = c(-Inf, Inf), y = single_bpnn_this_dataset_score, Single_ELM = factor(single_bpnn_this_dataset_score) )
  
  ggplot(current_dataset, aes(ensemble_size, score, colour=MODELS)) + 
    geom_point() + geom_line() +
    geom_line(aes(y = single_elm_this_dataset_score, colour = "single_elm")) + 
    geom_line(aes(y = single_bpnn_this_dataset_score, colour = "single_bpnn")) 
    
  png_name <- paste("type4_primary_task_dataset_", unique_dataset, ".png", sep="")
  png_path <- file.path(script.dir, "..", "png", png_name)
  ggsave(png_path, width = 40, height = 20, units = "cm")
}  