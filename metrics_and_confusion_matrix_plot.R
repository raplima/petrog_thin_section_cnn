library("dplyr")
library("tidyr")
library("ggplot2")
library("caret")

# set working directory before starting!


# read files:

basedir = "../Data/PP_mc_wb_test/"

# Create  vector
list_of_files <- c("InceptionV3_fine_tuned_thin_section_combined", 
                   "ResNet50_fine_tuned_thin_section_combined", 
                   "MobileNetV2_fine_tuned_thin_section_combined", 
                   "VGG19_fine_tuned_thin_section_combined")

# loop through files
for ( filename in list_of_files){ 

    modelname = strsplit(filename, "_")[[1]][1]
    
    datafile = paste(basedir, filename, ".csv", sep="")
    df = read.csv(file=datafile, header=TRUE)
    
    # set the same factors for both columns:
    df$PredLabel = factor(x=df$PredLabel,
                          levels=c("Argillaceous_siltstone",
                                   "Bioturbated_siltstone",
                                   "Massive_calcareous_siltstone",
                                   "Massive_calcite-cemented_siltstone",
                                   "Porous_calcareous_siltstone"),
                          labels=c("AS", 
                                   "BS",
                                   "MCS",
                                   "MCCS",
                                   "PCS"))
    
    df$TrueLabel = factor(x=df$TrueLabel,
                          levels=c("Argillaceous_siltstone",
                                   "Bioturbated_siltstone",
                                   "Massive_calcareous_siltstone",
                                   "Massive_calcite-cemented_siltstone",
                                   "Porous_calcareous_siltstone"),
                          labels=c("AS", 
                                   "BS",
                                   "MCS",
                                   "MCCS",
                                   "PCS"))
    
    # use caret to compute confusion matrix and other statistics:
    conf_matrix = confusionMatrix(data = df$PredLabel, reference = df$TrueLabel)
    
    # prepare to plot confusion matrix 
    # (https://stackoverflow.com/questions/7421503/how-to-plot-a-confusion-matrix-using-heatmaps-in-r)
    cm_long <- as.data.frame(conf_matrix$table)
    
    # change "y" order
    cm_long <- cm_long %>%
          mutate(Reference = factor(Reference), # alphabetical order by default
                 Prediction = factor(Prediction, levels = rev(unique(Prediction)))) # force reverse alphabetical order
    
    fig_title = paste(modelname, " - Acc: ", 
                      round(conf_matrix$overall[1], 2), 
                      ", Kappa: ", round(conf_matrix$overall[2],2), 
                      sep="")
    
    ggplot(cm_long, aes(x=Reference, y=Prediction, fill=Freq)) +
      geom_tile() + theme_bw() + coord_equal() + 
      scale_fill_distiller(palette="Greens", direction=1) +
      guides(fill=F) + # removing legend for `fill`
      labs(title = fig_title) +
      geom_text(aes(label=Freq), color="black")
      
    ggsave(paste(basedir, filename, "-confusion_matrix.pdf", sep=""), width = 4.25, height = 4)
}