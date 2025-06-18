# Set up directories and add data from Harvard Dataverse repository
args <- commandArgs(trailingOnly = TRUE)

get_gpt <- args[1]

library(dataverse)
library(data.table)

Sys.setenv("DATAVERSE_SERVER" = "dataverse.harvard.edu")

doi_data      <- "10.7910/DVN/OOSYCN"
doi_estimates <- "10.7910/DVN/QPU9GL"

# Create directories.
dirs <- c("./data/raw/supplement",
          "./data/raw/kawintiranon_2021",
          "./data/raw/li_2021",
          "./data/processed", 
          "./data/training", 
          "./data/results/summary_statistics",
          "./paper/latex", 
          "./figs")

sapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE)

# Pull raw and supplement data from dataverse:
get_data <- function(file_info, doi_link){
  
  # Get directory label
  filepath <- paste0("./data/", file_info['directoryLabel'], "/", file_info['originalFileName'])
  
  # Download the file
  print(sprintf("Getting file: %s", filepath))
  data <- get_dataframe_by_name(filename = file_info['filename'], dataset = doi_link, 
                                server = "dataverse.harvard.edu", original = T, .f = fread)
  
  write.csv(data, filepath, row.names = F)
  
  # Try not to hammer the Harvard Dataverse with requests:
  sys.sleep(5)
  
  return("Finished!")
  
}   

raw <- get_dataset(doi_data, format = "original", version = ":latest")$files

apply(raw, 1, get_data, doi_link = doi_data)


# If GPT models are not being estimated, getting existing estimates from dataverse:

if (get_gpt){
  
  # Only hold onto estimates concerning GPT:
  gpt <- get_dataset(doi_estimates, format = "original", version = ":latest")$files
  gpt <- gpt[grepl("gpt", gpt$label),]
  
  apply(gpt, 1, get_data, doi_link = doi_estimates)
  
}

