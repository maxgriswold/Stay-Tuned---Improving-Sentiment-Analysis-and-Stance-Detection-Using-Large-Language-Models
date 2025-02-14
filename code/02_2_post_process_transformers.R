# Post process transformer results
# Max Griswold
# 3/26/24

library(data.table)

setwd("C:/Users/griswold/Documents/github/twitter-representative-pop/public_facing/data/interim")

pol <- T

if (pol){
  pop <- "pol"
}else{
  pop <- "user"
}

pretrained <- fread(list.files(full.names = T)[(list.files() %like% "zero") & (list.files() %like% pop)])
tuned      <- list.files(full.names = T)[(list.files() %like% "tuned") & (list.files() %like% pop)]

tuned <- rbindlist(lapply(tuned, fread))
tuned[, method := paste0(method, "-tuned")]

tuned <- tuned[, .(id, method, sentiment_tweet)]

setnames(pretrained, c("model_name", "twitter_sentiment"), c("method", "sentiment_tweet"))
pretrained[, method := paste0(method, "-pretrained")]

pretrained <- pretrained[, .(id, method, sentiment_tweet)]

df <- rbind(tuned, pretrained)

write.csv(df, sprintf("sentiment_score_%s_transformer.csv", pop), row.names = F)


