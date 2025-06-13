# Set up directories and add data from Harvard Dataverse repository

setwd("C:/Users/Griz/Documents/GitHub/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models/")


library(dataverse)


Sys.setenv("DATAVERSE_KEY" = "0e65d139-9a2c-4fab-bd5f-aaea8edc52c8")

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