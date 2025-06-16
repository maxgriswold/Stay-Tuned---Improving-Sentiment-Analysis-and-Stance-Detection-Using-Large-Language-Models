# Estimate dictionary-based sentiment scores

rm(list = ls())

library(sentimentr)
library(tm)
library(qdap)
library(lexicon)
library(vader)

library(data.table)
library(plyr)
library(dplyr)
library(stringi)

df_li   <- fread("./data/processed/li_tweets_processed.csv")
df_kaw  <- fread("./data//processed/kawintiranon_tweets_processed.csv")
df_pol  <- fread("./data//processed/pol_tweets_processed.csv")
df_user <- fread("./data//processed/user_tweets_processed.csv")

analysis_datasets <- list("pol" = df_pol, 
                          "user" = df_user,
                          "li_2021" = df_li,
                          "kawintiranon_2021" = df_kaw)

# Set up lexical dictionaries:
lexnames <- c("huliu", "nrc", "senticnet", "socal_google", "sentiword")

dicts <- list(hash_sentiment_huliu, 
              hash_sentiment_nrc,
              hash_sentiment_senticnet, 
              hash_sentiment_socal_google, 
              hash_sentiment_sentiword)

names(dicts) <- lexnames

run_lex_model <- function(lexname, df){
  
  print(sprintf("Starting %s", lexname))
  score <- df %>%
           get_sentences(.$text) %>%
           sentiment(polarity_dt = dicts[[lexname]],
                     amplifier.weight = 0.2,
                     n.before = 0,
                     n.after = 2,
                     adversative.weight = 0)
  
  # Collapse sentiment scores back to tweet-level:
  score <- score[!is.na(word_count),]
  
  score[, (lexname) := weighted.mean(.SD$sentiment, w = .SD$word_count), by = "element_id"]
  
  score <- unique(score[, c("id", "element_id", lexname), with = F])
  
  return(score)
  
}

run_lexical_pipeline <- function(df_name){
  
  dd <- analysis_datasets[[df_name]]
  dd <- dd[, .(id, text)]
  
  # Remove a few special characters to ensure VADER works appropriately:
  dd[, text := gsub("\\&|\\-", "", text)]
  
  res  <- Reduce(function(x, y) merge(x, y, by = c("id", "element_id")), 
                 lapply(names(dicts), run_lex_model, df = dd))
  
  # Pull out VADER scores and set manually within results dataframe
  res_vader <- vader_df(dd$text)
  res_vader <- res_vader$compound
  
  res_temp <- copy(dd[, .(id)])
  res_temp[, vader := res_vader]
  
  df_score <- join(dd, res, by = c("id"), type = 'left')
  df_score <- join(df_score, res_temp, by = c("id"))
  
  df_score <- melt(df_score, id.vars = c("id"),
                   measure.vars = c(lexnames, "vader"), variable.name = 'method',
                   value.name = 'sentiment_tweet')
  
  write.csv(df_score, sprintf("./data/results/lexical_%s.csv", df_name), row.names = F)
  
}

for (df_name in names(analysis_datasets)){
  print(sprintf("Running lexical models for %s", df_name))
  run_lexical_pipeline(df_name)
}

