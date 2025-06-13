# Set up directories and add data from Harvard Dataverse repository

library(dataverse)

doi <- "doi:10.7910/DVN/OOSYCN"

# Add additional folders, if they don't already exist

if (!dir.exists("./data")){
  dir.create("./data")
}

if (!dir.exists("./data/processed")){
  dir.create("./data/processed")
}

if (!dir.exists("./data/interim")){
  dir.create("./data/interim")
}

if (!dir.exists("./data/training")){
  dir.create("./data/training")
}

if (!dir.exists("./data/results")){
  dir.create("./data/results")
}