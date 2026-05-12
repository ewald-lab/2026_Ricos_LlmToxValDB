setwd("/Users/jewald/Desktop/IVIVE")


library(httk)

allData <- read.csv("./httk_input.csv")

## In order for values to intergrate correctly we need to ensure they are character type
allData$FUB_pred <- as.character(allData$FUB_pred)
allData$Clint_pred <- as.character(allData$Clint_pred)
allData$LogP_pred <- as.character(allData$LogP_pred)
allData$AVERAGE_MASS <- as.character(allData$AVERAGE_MASS)
allData$Chemical.Class <- ""

chem.physical_and_invitro.data <- add_chemtable(
  allData,
  current.table = chem.physical_and_invitro.data,
  data.list = list(Compound = "Compound_name",
                   CAS = "runID", #OK just using as an identifier
                   MW = "AVERAGE_MASS",
                   logP = "LogP_pred",
                   Funbound.plasma = "FUB_pred",
                   Clint = "Clint_pred",
                   Chemical.Class = "Chemical.Class"
  ),
  species = "Human",
  reference = "OPERA")

# Doesn't update parameters correctly which causes errors, so manually remove old data
chem.physical_and_invitro.data <- chem.physical_and_invitro.data[grepl("runID", chem.physical_and_invitro.data$CAS), ]

# Prepare df for results
allData$POD_um_50 <- NA
allData$POD_um_95 <- NA
allData$POD_um_l_50 <- NA
allData$POD_um_l_95 <- NA
allData$POD_um_u_50 <- NA
allData$POD_um_u_95 <- NA

inputData <- allData

for(n in c(1:nrow(inputData))){
  runID <- inputData$runID[n]
  print(runID)
  
  ## Calculate POD_um
  result <- calc_mc_oral_equiv(inputData$POD_um[n],
                               chem.cas = runID,
                               which.quantile = c(0.50, 0.95),
                               species = "Human",
                               input.units = "uM",
                               output.units = "mgpkgpday",
                               restrictive.clearance = TRUE,
                               model = "3compartmentss")
  
  result_l <- calc_mc_oral_equiv(inputData$POD_um_l[n],
                                 chem.cas = runID,
                                 which.quantile = c(0.50, 0.95),
                                 species = "Human",
                                 input.units = "uM",
                                 output.units = "mgpkgpday",
                                 restrictive.clearance = TRUE,
                                 model = "3compartmentss")
  
  result_u <- calc_mc_oral_equiv(inputData$POD_um_u[n],
                                 chem.cas = runID,
                                 which.quantile = c(0.50, 0.95),
                                 species = "Human",
                                 input.units = "uM",
                                 output.units = "mgpkgpday",
                                 restrictive.clearance = TRUE,
                                 model = "3compartmentss")
inputData$POD_um_50[
  inputData$runID == runID] <- result[1]
inputData$POD_um_95[
  inputData$runID == runID] <- result[2]
inputData$POD_um_l_50[
  inputData$runID == runID] <- result[1]
inputData$POD_um_l_95[
  inputData$runID == runID] <- result[2]
inputData$POD_um_u_50[
  inputData$runID == runID] <- result[1]
inputData$POD_um_u_95[
  inputData$runID == runID] <- result[2]
}

write.csv(inputData, file = "httk_output.csv")

