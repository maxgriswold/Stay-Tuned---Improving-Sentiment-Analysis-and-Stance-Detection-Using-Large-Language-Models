# Process external datasets for validating models:
# Max Griswold
# 11/20/2024

library(data.table)
library(plyr)

setwd("C:/users/griswold/documents/GitHub/twitter-representative-pop/public_facing/data/")

li_train_trump <- fread("./raw/li_2021/raw_train_trump.csv")
li_train_biden <- fread("./raw/li_2021/raw_train_biden.csv")

df_li <- rbind(li_train_trump, li_train_biden)
setnames(df_li, names(df_li), c("text", "subject", "score"))

df_li[, subject := ifelse(subject == "Donald Trump", "trump", "biden")]
df_li[, score := ifelse(score == "AGAINST", -1, 1)]
df_li[, id := .I]

write.csv(df_li, "./processed/li_tweets_processed.csv", row.names = F)
  
kaw_train_trump <- fread("./raw/kawintiranon_2021/trump_stance_train_public.csv")
kaw_test_trump  <- fread("./raw/kawintiranon_2021/trump_stance_test_public.csv")

kaw_trump <- rbind(kaw_train_trump, kaw_test_trump)
kaw_trump$subject <- "trump"

kaw_train_biden <- fread("./raw/kawintiranon_2021/biden_stance_train_public.csv")
kaw_test_biden <- fread("./raw/kawintiranon_2021/biden_stance_test_public.csv")

kaw_biden <- rbind(kaw_train_biden, kaw_test_biden)
kaw_biden$subject <- "biden"

df_kaw <- setDT(rbind(kaw_biden, kaw_trump))
df_kaw[, label := mapvalues(label, c("AGAINST", "NONE", "FAVOR"), c(-1, 0, 1))]
df_kaw <- df_kaw[, .(text, label, subject)]
setnames(df_kaw, names(df_kaw), c("text", "score", "subject"))

df_kaw[, id := .I]

write.csv(df_kaw, "./processed/kawintiranon_tweets_processed.csv", row.names = F)
