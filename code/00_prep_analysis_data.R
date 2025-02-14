# Prep analysis dataset for twitter sentiment project
# Max Griswold

rm(list = ls())

library(mongolite)

library(data.table)
library(plyr)
library(dplyr)
library(stringi)
library(lubridate)

setwd("C:/users/griswold/documents/GitHub/twitter-representative-pop/")

# Connect to the Mongo database and pull tweets which contain the word
# "Biden" between presidential debates:

con <- mongo(collection = "politician_tweets_bw", db = "twitter",
             url = "mongodb://root:password@twitter-collection-db.rand.cloud",
             verbose = F)

con_prof <- mongo(collection = "politician_profiles", db = "twitter",
                  url = "mongodb://root:password@twitter-collection-db.rand.cloud",
                  verbose = F)

# We need to pass a "find" command to the mongo database to pull data. To do
# so, we need to specify the documents we would like to pull with a query
# and the fields we would like to see returned.

qb <- '{"created_at" : {"$exists": true}, "text" : {"$regex" : "^(?=.*biden)(?:(?!trump).)*$", "$options" : "i"}}'
qt <- '{"created_at" : {"$exists": true}, "text" : {"$regex" : "^(?=.*trump)(?:(?!biden).)*$", "$options" : "i"}}'

qprof <- '{"created_at" : {"$exists": true}}'

# "1" below indicates we want the field maintained and pulled; "0" indicates the
# field should be explicitly dropped

f <- '{"created_at":1, "text":1, "_id":0, "author_id":1, "id":1}'

fprof <- '{"id":1, "name":1, "_id":0, "username":1}'

# The limit argument sets a maximum amount of observations to return (useful
# for debugging)

biden <- con$find(query = qb, fields = f)
trump <- con$find(query = qt, fields = f)

profiles <- con_prof$find(query = qprof, fields = fprof)

biden$subject <- "biden"
trump$subject <- "trump"

df <- rbind(biden, trump)
setDT(df)

write.csv(df, "./public_facing/data/raw/politician_tweets.csv", row.names = F)

# Convert created_at into date columns, then subset data to dates 
# seven days before the first debate and seven days after the election

key_dates <- data.table("date" = as.Date(c("29/09/2020",
                                           "07/10/2020","22/10/2020", 
                                           "03/11/2020"), "%d/%m/%y"),
                        "date_label" = c( "First\ndebate", "VP\ndebate",
                                          "Second\ndebate", "Election\nday"))

df[, date := as.Date(created_at, format = "%Y-%m-%d")]
df[, created_at := NULL]

# Convert text into sentences
df_original <- copy(df)

# Remove hyperlinks and html
df[, text := gsub("http\\S+", "", text)]
df[, text := gsub("/t.co\\S+", "", text)]
df[, text := gsub("<.*>", "", text)]

df[, text := gsub("[^\x01-\x7F]", "", text)]

# Get politician names and add onto tweets:
setnames(profiles, names(profiles), c("name", "author_id", "username"))
df <- join(df, profiles, by = 'author_id', type = 'left')

# Grab info on pols from VoteView scores + Robbins files:
lean <- fread("pol_lean.csv") 

reps <- fread("RepsTwitter2022.csv")
setnames(reps, names(reps), c("first", "last", "username", "loc", "party"))

sens <- fread("SensTwitter2022.csv")
setnames(sens, names(sens), c("loc", "party", "first", "last", "username"))

pol_names <- rbind(sens, reps)
pol_names[, username := gsub("@", "", username)]

df <- join(df, pol_names, by = "username", type = "left")

# Remove Governors and former governers (confirmed by checking list, hence carve out for Mark Pocan)
df <- df[!is.na(last)|name == "Rep. Mark Pocan", ]
df[is.na(last), `:=`(first = "Mark", last = "Pocan")]

df[, bioname := paste(first, last)]

# Convert string encoding to ensure accents are included. Set lower to improve fuzzy
# match with VoteView
df[, bioname := iconv(bioname, from = "latin1", to='ASCII//TRANSLIT')]
df[, bioname := tolower(bioname)]

# David Cicilline tweets out of two accounts so make sure tweets on the second account are 
# maintained

df[username == "RepCicilline", `:=`(last = "Cicilline", first = "David", bioname = "David Cicilline")]

# See VoteView paper concerning nominate factor: https://link.springer.com/article/10.1007/s11127-018-0546-0
# (first factor dimension appears to capture liberal/conservative lean of members)
lean <- lean[, .(bioname, nominate_dim1, party_code)]

lean[, party_code := as.character(party_code)]

lean[party_code == "100", party_code := "D"]
lean[party_code == "200", party_code := "R"]

lean[bioname %in% c("AMASH, Justin", "MITCHELL, Paul"), party_code := "R"]
lean[bioname %in% c("SANDERS, Bernard", "KING, Angus Stanley, Jr."), party_code := "D"]

# Change bioname to be first/last rather than last/first:
lean[, c("last", "first") := tstrsplit(bioname, " ", keep = c(1, 2))]
lean[, bioname := paste(first, last)]
lean[, bioname := tolower(gsub(",", "", bioname))]

# Changing the encoding a bit, compared to earlier, since accents are being
# displayed directly (rather than incorporated as backslash special characters)
lean[, bioname := iconv(bioname, to = 'ASCII//TRANSLIT')]

lean <- lean[, .(bioname, party_code, nominate_dim1)]

#Try direct match
df <- join(df, lean, by = "bioname", type = "left")

# For stragglers, try using only last names:
df_stragglers <- df[is.na(party_code)]
df_stragglers[, last := iconv(last, from = "latin1", to='ASCII//TRANSLIT')]
df_stragglers[, last := tolower(last)]

df_stragglers[, `:=`(bioname = NULL, party_code = NULL, nominate_dim1 = NULL)]

lean[, last := tstrsplit(bioname, " ", keep = 2)]

df_stragglers <- join(df_stragglers, lean, by = "last", type = "left")

# For remainder, drop these reps since they are not included in VoteView 
# (since they are junior members/senators w/o voting records)

# We lose ~3% of observations here (361/11489 tweets)
df <- rbind(df[!is.na(party_code)], df_stragglers[!is.na(party_code)], fill = T)
df <- df[, .(id, author_id, bioname, text, subject, date, nominate_dim1, party_code, loc)]

# Additionally, remove empty tweets:
df <- df[text != ""]

# Construct new ID column to make sure we maintain the right number
# of observations across models
df[, id := .I]

df <- df[, .(id, bioname, text, subject, date, nominate_dim1, party_code, loc)]

write.csv(df, "./public_facing/data/processed/pol_tweets_processed.csv", row.names = F)

# Process hand-coded validation tweet dataset to match format of 
# politican tweets
df_biden <- fread("Biden_tweets_for_scoring.csv")
df_biden[, subject := "biden"]

df_trump <- fread("Trump_tweets_for_scoring.csv")
df_trump[, subject := "trump"]

df_original <- rbindlist(list(df_trump, df_biden))
setnames(df_original, names(df_original), c("tweet_id", "date", "username", "text", "subject"))

df_original <- df_original[, .(tweet_id, date, username, text)]

# Convert date to same time zone as the other dataset (thanks excel).
df_original[, date := as.POSIXct(date, tz="UTC")]
df_original[, date := format(date, "%m/%d/%Y %H:%M")]

df <- fread("pol_twitter_scored.csv")

# Replace text in Robbin's scored tweets w/ fixed text in original tweets:
setnames(df, names(df), c("date", "username", "subject", "scorer_1", "scorer_2"))
df <- df[, .(date, username, subject, scorer_1, scorer_2)]

# Average hand-code together to get final score:
df[, score := (scorer_1 + scorer_2)/2]

# Standardizing date/time due to errors introduced by excel:
df[, date := as.POSIXct(date, format = "%m/%d/%Y %H:%M")]
df[, date := format(date, "%m/%d/%Y %H:%M")]

df <- join(df, df_original, by = c("date", "username"), type = "left")
write.csv(df, "./public_facing/data/raw/user_tweets.csv", row.names = F)

# Remove hyperlinks and html,
# (future model iterations should likely code these UTF-8 hashes w/ sentiment
# scores directly); then remove any lingering html.
df[, text := gsub("http\\S+", "", text)]
df[, text := gsub("/t.co\\S+", "", text)]
df[, text := gsub("<.*>", "", text)]
df[, text := gsub("[^\x01-\x7F]", "", text)]

# Create author_id based on person name
df[, author_id := as.numeric(as.factor(username))]

# Recode id to make it easier to merge later; given the length of the id (integer64),
# there have been issues with merges across steps:
df[, id := .I]

write.csv(df, "./public_facing/data/processed/user_tweets_processed.csv", row.names = F)

# Similarly, prep validation tweets
df_biden_val <- fread("./public_facing/data/raw/biden_user_train.csv")
df_biden_val[, subject := "biden"]

df_trump_val <- fread("./public_facing/data/raw/trump_user_train.csv")
df_trump_val[, subject := "trump"]

df_val <- rbindlist(list(df_trump_val, df_biden_val))

# Fix the excel issue from the underlying sheet.
setnames(df_val, names(df_val), c("tweet_id", "date", "username", "scorer_1", "scorer_2", "full_text", "subject"))

# Average hand-code together to get final score:
df_val[, score := (scorer_1 + scorer_2)/2]

# Standardizing date/time due to errors introduced by excel:
df_val[, date := as.POSIXct(date, format = "%m/%d/%Y %H:%M")]
df_val[, date := format(date, "%m/%d/%Y %H:%M")]

df_val <- join(df_val, df_original, by = c("date", "username"), type = "left")
df_val <- df_val[, .(date, username, scorer_1, scorer_2, score, subject, text)]

write.csv(df_val, "./public_facing/data/raw/user_handcode_train.csv", row.names = F)

df_val[, text := gsub("http\\S+", "", text)]
df_val[, text := gsub("/t.co\\S+", "", text)]
df_val[, text := gsub("<.*>", "", text)]
df_val[, text := gsub("[^\x01-\x7F]", "", text)]

df_val[, author_id := as.numeric(as.factor(username))]
df_val[, id := .I]

df_val <- df_val[, .(author_id, id, text, score, subject)]
write.csv(df_val, "./public_facing/data/processed/handcode_tweets_processed.csv", row.names = F)
