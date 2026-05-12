
setwd("/Users/jewald/Desktop/IVIVE")

library(readxl)
library(dplyr)

OASIS_PODs <- read_xlsx("OASIS_PHH_Dino_PODS.xlsx")
DTXSIDtoOASIS <- read.csv("matchDTXSIDtoOASIS.csv")
OPERApredictions <- read.csv("Needed_OPERA_predictions.csv")
molecularWeight <- read.csv("CCD-Batch-Search_2025-06-17_01_52_31.csv")
molecularWeight <- molecularWeight[,c("DTXSID", "AVERAGE_MASS")]
molecularWeight$AVERAGE_MASS <- as.numeric(molecularWeight$AVERAGE_MASS)
molecularWeight <- molecularWeight[!(is.na(molecularWeight$AVERAGE_MASS)),]

## Remove where no OASIS ID is availible
OASIS_PODs <- OASIS_PODs[!(is.na(OASIS_PODs$OASIS_ID)),]

## Remove where OPERA predictions could not be made
OPERApredictions <- OPERApredictions[complete.cases(OPERApredictions),]

## Merge OASIS IDs to DTXSIDs
allData <- merge(x=DTXSIDtoOASIS, y=OASIS_PODs, by = "OASIS_ID")
allData <- merge(x=allData, y=OPERApredictions,
                 by.x = "DTXSID", by.y = "MoleculeID")
allData <- merge(x=allData, y=molecularWeight, by = "DTXSID")
allData <- distinct(allData)

allData$runID <- paste0("runID_", c(1:nrow(allData)))

write.csv(allData, "./httk_input.csv", row.names = FALSE)