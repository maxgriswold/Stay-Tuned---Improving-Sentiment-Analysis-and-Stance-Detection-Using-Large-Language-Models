# Prep training datasets for LLMs

rm(list = ls())

set.seed(2985671)

library(httr)
library(jsonlite)
library(data.table)
library(plyr)
library(dplyr)

df_pol  <- fread("./data/processed/pol_tweets_processed.csv")
df_user <- fread("./data/processed/handcode_tweets_processed.csv")

prep_train_data <- function(dataset, subject_name, formatting = 'llama'){
  
  # For pol-based datasets, set score based off dataset name (e.g., if Biden, dem +1, rep -1) 
  # and subset data to prespecified training rows (a randomly selected fold)
  # Otherwise, use the prepared hand-coded validation dataset from the user database.
  
  # write.csv(df_train , sprintf("./data/interim/training_key_%s_user.csv", subject_name))
  if (dataset != "handcode"){
    
    df_train <- fread(sprintf("./data/interim/training_key_%s.csv", subject_name))
    dd <- df_pol[id %in% df_train[train == T,]$id,]
    
    if (subject_name == "biden" & dataset == "party_id"){
      dd[, score := ifelse(party_code == "D", 1, -1)]
    }else if (subject_name == "trump" & dataset == "party_id"){
      dd[, score := ifelse(party_code == "D", -1, 1)]
    }
    
    # Reverse code first-dimension of nominate to correspond to view towards
    # specific party. So democrats have a "positive" nominate score for Biden and
    # "negative" nominate score for Trump
    if (subject_name == "biden" & dataset == "nominate"){
      dd[, score := nominate_dim1*-1]
    }else if (subject_name == "trump" & dataset == "nominate"){
      dd[, score := nominate_dim1]
    }
  }else{
    df_train <- fread(sprintf("./data/interim/training_key_%s_handcode.csv", subject_name))
    dd <- df_user[id %in% df_train[train == T,]$id,]
  }
  
  train_json <- create_json(dd, dataset, subject_name, formatting)
  
  # Keeping in case this is useful for other researchers who may wish
  # to train LLama models in the future
  
  if (formatting == 'llama'){
    train_json <- toJSON(train_json, pretty = T, auto_unbox = T)
    
    output_file <- sprintf("./data/training/llm_finetune_llama_%s_%s.json", dataset, subject_name)
    
    write(train_json, output_file)
  }else{
    output_file <- sprintf("./data/training/llm_finetune_openai_%s_%s.jsonl", dataset, subject_name)
    writeLines(
      sapply(train_json, function(x) toJSON(x, auto_unbox = TRUE)),
      con = output_file
    )
  }

  return()
  
}

create_json <- function(dd, dataset, subject_name, formatting = 'llama' ){
  
  if (dataset == 'party_id'){
    user <- sprintf('Provide a binary-value score that is either -1 or 1 that determines whether the author of the text likes or dislikes %s, with -1 indicating dislike and 1 indicating like. Put the response in the following format, replacing the curly brackets with the score value. Score:{score} \n %s', dd$subject, dd$text)
  }else{
    user <- sprintf('Provide a score between -1 and 1 that determines whether the author of the text likes or dislikes %s, with -1 indicating greatest dislike and 1 indicating greatest like. Put the response in the following format, replacing the curly brackets with the score value. Score:{score} \n %s', dd$subject, dd$text)
    
  }

  assistant <-  sprintf("Score: %s", dd$score)
  
  df_message <- data.frame("user" = user, "assistant" = assistant)
  
  # Prepare the JSON structure
  output <- lapply(1:nrow(df_message), function(i) {
    list(messages = list(
      list(role = "user", content = df_message$user[i]),
      list(role = "assistant", content = df_message$assistant[i])
    ))
  })
  
  return(output)
  
}

dataset_names <- c("party_id", "nominate", "handcode")
subject_names <- c("biden", "trump")
format_names    <- c("openai")

for (d in dataset_names){
  for (s in subject_names){
    for (form in format_names)
    prep_train_data(d, s, form)
  }
}