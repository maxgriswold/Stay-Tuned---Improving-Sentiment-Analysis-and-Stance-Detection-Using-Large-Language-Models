# Combine results into final dataframe

library(data.table)
library(plyr)

path  <- "./data/results"
files <- list.files(path)

# Lexical models
lexical_models <- files[files %like% "lexical"]

prep_lexical <- function(f){
  
  dd <- fread(sprintf("%s/%s", path, f))
  data_name <- gsub("lexical_|.csv", "", f)
  
  dd[, data_name := data_name]
  dd[, tuned := F]
  dd[, tune_data := NA]
  dd[, prompt_name := NA]
  
  setnames(dd, "method", "model_name")
  
  return(dd)
  
}

df_lexical <- rbindlist(lapply(lexical_models, prep_lexical))

# Pretrained semi-supervised models
df_semi_pretrained <- fread(sprintf("./%s/zero_shot_results.csv", path))

df_semi_pretrained[, tuned := F]
df_semi_pretrained[, tune_data := NA]
df_semi_pretrained[, prompt_name := NA]
df_semi_pretrained[, subject := NULL]

# Tuned semi-supervised models
semi_tuned_models <- files[grep("tweetnlp|siebert|distilbert|deberta", files)]

prep_semi_tuned <- function(f){
  
  dd <- fread(sprintf("./%s/%s", path, f))
  model_name <- gsub("_tune.*", "", f)
  tune_data <- gsub(".*_(party|handcode|nominate)_.*", "\\1", f)
  data_name <- gsub(".*_(kawintiranon|li|pol|user_val)_.*", "\\1", f)
  
  if (model_name =='deberta' & tune_data == 'handcode'){
    dd[, sentiment_tweet := sentiment_tweet*-1]
  }
  
  dd[, tuned := T]
  dd[, tune_data := tune_data]
  dd[, data_name := data_name]
  dd[, model_name := model_name]
  dd[, prompt_name := NA]
  
  return(dd)
  
}

df_semi_tuned <- rbindlist(lapply(semi_tuned_models, prep_semi_tuned))

# GPT models
gpt_models <- files[(files %like% "gpt")]

prep_gpt <- function(f){
  
  dd <- fread(sprintf("./%s/%s", path, f))
  
  model_name <- gsub("(gpt35|gpt4|gpt4o).*", "\\1", f)
  tuned <- ifelse(grepl("tune", f) == T, T, F)
  tune_data <- ifelse(tuned, gsub(".*_(party|handcode|nominate)_.*", "\\1", f), NA)
  data_name <- gsub(".*_(kawintiranon|li|pol|user_val)_.*", "\\1", f)
  prompt_name <- gsub(".*_(p[0-9]+)_.*", "\\1", f)
  
  dd[, tuned := tuned]
  dd[, tune_data := tune_data]
  dd[, data_name := data_name]
  dd[, model_name := model_name]
  dd[, prompt_name := prompt_name]
  
  return(dd)
  
}

df_gpt <- rbindlist(lapply(gpt_models, prep_gpt))

# Combine all results
df_final <- rbindlist(list(df_gpt, df_semi_tuned, df_semi_pretrained, df_lexical), use.names = T)
df_final <- df_final[, .(model_name, tuned, tune_data, data_name, prompt_name, id, sentiment_tweet)]

# Resolve inconsistency in data_name column:
df_final[, data_name := mapvalues(data_name, from = c("kawintiranon_2021", "li_2021"), to = c("kawintiranon", "li"))]

# Add on observed data:
merge_scores <- function(d_name){
  
  df_score <- fread(sprintf("./data/processed/%s_tweets_processed.csv", d_name))
  df_sub <- df_final[data_name == d_name,]
  
  if (d_name == 'pol'){
    df_score[subject == "biden", score := ifelse(party_code == "D", 1, -1)]
    df_score[subject == "trump", score := ifelse(party_code == "D", -1, 1)]
    
    df_score[, score_nominate := ifelse(subject == "biden", nominate_dim1*-1, nominate_dim1), ]
    
  }else{
    df_score[, score_nominate := NA]
  }
  
  # If merging user data, determine rows where text mentions both subjects.
  # Otherwise, set column for both_subjects equal to F:
  if (d_name == 'user_val'){

    biden_and_trump <- "(?i)(?=.*biden)(?=.*trump)"
    df_score[, both_subjects := ifelse(grepl(biden_and_trump, text, perl = TRUE), T, F)]
    
  }else{
    df_score[, both_subjects := F]
  }
  
  df_score <- df_score[, .(id, score, score_nominate, subject, both_subjects)]
  df_sub <- join(df_sub, df_score, by = 'id', type = 'left')
  
  return(df_sub)
  
}

data_names <- unique(df_final$data_name)
df_final <- rbindlist(lapply(data_names, merge_scores))

df_final[, model_id := as.numeric(factor(paste0(model_name, tune_data, data_name, prompt_name, subject)))]
df_final <- df_final[, .(model_id, model_name, tuned, tune_data, data_name, prompt_name, subject, both_subjects, id, sentiment_tweet, score, score_nominate)]

setnames(df_final, c("sentiment_tweet"), c("est_score"))

# For the small number of GPT models where responses returned text indicating GPT would
# not assign a score due to a lack of stance, set these values to 0 This affects
# 4 not-tuned GPT 3.5 models using prompt 6, where 1 - 4% of estimates were not numeric values.
# Overall, less than 0.09% of estimated score values impacted.

df_final[is.na(est_score), est_score := 0]

# Remove a few odd duplicates that occurred (~90 rows/1.5 million)
df_final <- unique(df_final)
df_final <- df_final[, .SD[1], by = c("model_id", "id")]

square_data <- function(m_id){
  
  dd <- df_final[model_id == m_id, ]

  d_name    <- unique(dd$data_name)
  s_name <- unique(dd$subject)
  
  df_score <- fread(sprintf("./data/processed/%s_tweets_processed.csv", d_name))
  
  if (d_name == 'pol'){
    df_score[subject == "biden", score := ifelse(party_code == "D", 1, -1)]
    df_score[subject == "trump", score := ifelse(party_code == "D", -1, 1)]
  }
  
  if (d_name == 'user_val'){
    
    biden_and_trump <- "(?i)(?=.*biden)(?=.*trump)"
    df_score[, both_subjects := ifelse(grepl(biden_and_trump, text, perl = TRUE), T, F)]
    
  }else{
    df_score[, both_subjects := F]
  }
  
  df_score <- df_score[subject == s_name, .(id, score, both_subjects)]
  
  dc <- setDT(join(dd, df_score, by = 'id', type = 'full'))
  
  dc[is.na(est_score), `:=`(model_id = unique(dd$model_id),
                            model_name = unique(dd$model_name),
                            tuned = unique(dd$tuned),
                            tune_data = unique(dd$tune_data),
                            data_name = unique(dd$data_name),
                            prompt_name = unique(dd$prompt_name),
                            subject = unique(dd$subject))]
  
  dc[is.na(est_score), est_score := 0]
  
}

df_final <- rbindlist(lapply(unique(df_final$model_id), square_data))

# Add on an indicator for whether the data was used for training or not:
df_score <- fread("./data/processed/pol_tweets_processed.csv")
train_id_biden <- fread("./data/raw/supplement/training_key_biden.csv")
train_id_trump <- fread("./data/raw/supplement/training_key_trump.csv")

train_id <- rbind(train_id_biden, train_id_trump)
train_id[, data_name := 'pol']

df_score <- df_score[id %in% train_id$id,]

df_final <- join(df_final, train_id, by = c('data_name', 'id'), type = "left")

# Add on hand-code scores for subset of politician data:
df_hand_pol <- fread("./data/raw/supplement/politician_tweets_handcoded.csv")[, .(id, score)]
df_hand_pol[, data_name := 'pol']
setnames(df_hand_pol, "score", "score_coded")

df_final <- join(df_final, df_hand_pol, by = c("id", "data_name"), type = "left")

write.csv(df_final, "./data/results/analysis_results.csv", row.names = F)
