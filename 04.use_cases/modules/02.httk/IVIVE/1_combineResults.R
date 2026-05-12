## Get List of Files
## Make sure to change the path to your folder.
setwd("/Users/jewald/Desktop/IVIVE")

library(dplyr)

folder <- "./opera_output/"
listOfFiles <- list.files(path=folder)

getFileFunction <- function(X, folder){
  read.csv(paste0(folder, X))
}

# Combine files together
combinedResults <- do.call(rbind, lapply(
  X=listOfFiles, folder = folder, FUN = getFileFunction))

write.csv(combinedResults, file = "ALL_OPERA_predictions.csv",
          row.names = FALSE)

neededParameters <- combinedResults[,c("MoleculeID", "LogP_pred", "FUB_pred",
                                       "Clint_pred")]
neededParameters <- neededParameters[complete.cases(neededParameters),]

# Check for duplicated values
dupIDs <- neededParameters$MoleculeID[duplicated(neededParameters$MoleculeID)] %>% unique()
dupParameters <- neededParameters[neededParameters$MoleculeID %in% dupIDs, ]

# Duplicated DTXSID have same ADME parameters, so can remove
neededParameters <- neededParameters %>% distinct()

write.csv(neededParameters, file = "Needed_OPERA_predictions.csv",
          row.names = FALSE)
