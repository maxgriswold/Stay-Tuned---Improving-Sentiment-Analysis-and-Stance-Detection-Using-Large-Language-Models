# Prep analysis dataset for the code pipeline

rm(list = ls())

library(data.table)
library(plyr)
library(dplyr)
library(stringi)
library(lubridate)

# Code below is kept for posterity; see main paper for details on how
# we obtained Twitter data. This data had been stored in a MongoDB.
# Please contact study authors for more details.
#
# Connect to the Mongo database and pull tweets which contain the word
# "Biden" between presidential debates:
#
#library(mongolite)
#
# con <- mongo(collection = "politician_tweets_bw", db = "twitter",
#              url = "mongodb://root:password@twitter-collection-db.rand.cloud",
#              verbose = F)
# 
# con_prof <- mongo(collection = "politician_profiles", db = "twitter",
#                   url = "mongodb://root:password@twitter-collection-db.rand.cloud",
#                   verbose = F)
# 
# # Query mongo DB getting all politician tweets that only reference a single subject.
# 
# qb <- '{"created_at" : {"$exists": true}, "text" : {"$regex" : "^(?=.*biden)(?:(?!trump).)*$", "$options" : "i"}}'
# qt <- '{"created_at" : {"$exists": true}, "text" : {"$regex" : "^(?=.*trump)(?:(?!biden).)*$", "$options" : "i"}}'
# 
# qprof <- '{"created_at" : {"$exists": true}}'
# 
# # "1" below indicates we want the field maintained and pulled; "0" indicates the
# # field should be explicitly dropped
# 
# f <- '{"created_at":1, "text":1, "_id":0, "author_id":1, "id":1}'
# 
# fprof <- '{"id":1, "name":1, "_id":0, "username":1}'
# 
# # The limit argument sets a maximum amount of observations to return (useful
# # for debugging)
# 
# biden <- con$find(query = qb, fields = f)
# trump <- con$find(query = qt, fields = f)
# 
# profiles <- con_prof$find(query = qprof, fields = fprof)
# 
# write.csv(biden, "./data/raw/politician_tweets_biden.csv", row.names = F)
# write.csv(trump, "./data/raw/politician_tweets_trump.csv", row.names = F)
# write.csv(profiles, "./data/raw/politician_twitter_profiles.csv", row.names = F)

biden <- fread("./data/raw/politician_tweets_biden.csv")
trump <- fread("./data/raw/politician_tweets_trump.csv")
profiles <- fread("./data/raw/politician_twitter_profiles.csv")

biden$subject <- "biden"
trump$subject <- "trump"

df <- rbind(biden, trump)
setDT(df)

# Get politician names and add onto tweets:
setnames(profiles, names(profiles), c("name", "author_id", "username"))
df <- join(df, profiles, by = 'author_id', type = 'left')

df[, date := as.Date(created_at, format = "%Y-%m-%d")]
df[, created_at := NULL]

# Remove hyperlinks and html
df[, text := gsub("http\\S+", "", text)]
df[, text := gsub("/t.co\\S+", "", text)]
df[, text := gsub("<.*>", "", text)]
df[, text := gsub("[^\x01-\x7F]", "", text)]

# Grab info on pols from VoteView and Twitter username/pol crosswalks:
lean <- fread("./data/raw/supplement/pol_lean.csv")

reps <- fread("./data/raw/supplement/cw_house_2022.csv")
setnames(reps, names(reps), c("first", "last", "username", "loc", "party"))

sens <- fread("./data/raw/supplement/cw_senate_2022.csv")
setnames(sens, names(sens), c("loc", "party", "first", "last", "username"))

pol_names <- rbind(sens, reps)

# Make sure encoding is correct, which removes unexpected symbols from MongoDB
pol_names[, username := iconv(username, from = "ISO-8859-1", to = "UTF-8")]
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
# (first factor represents liberal/conservative lean of members)
lean <- lean[, .(bioname, nominate_dim1, party_code)]

lean[, party_code := as.character(party_code)]

lean[party_code == "100", party_code := "D"]
lean[party_code == "200", party_code := "R"]

lean[bioname %in% c("AMASH, Justin", "MITCHELL, Paul"), party_code := "R"]
lean[bioname %in% c("SANDERS, Bernard", "KING, Angus Stanley, Jr."), party_code := "D"]

# Change bioname to be first name/last rather than last/first:
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

write.csv(df, "./data/processed/pol_tweets_processed.csv", row.names = F)

# Read in the hand-coded train tweets
df <- fread("./data/raw/user_train_tweets.csv")

# Average hand-code together to get final score:
df[, score := (scorer_1 + scorer_2)/2]

# Remove hyperlinks and html,
df[, text := gsub("http\\S+", "", text)]
df[, text := gsub("/t.co\\S+", "", text)]
df[, text := gsub("<.*>", "", text)]
df[, text := gsub("[^\x01-\x7F]", "", text)]

# Create author_id based on person name
df[, author_id := as.numeric(as.factor(username))]

# Recode id to make it easier to merge later
df[, id := .I]

write.csv(df, "./data/processed/user_train_tweets_processed.csv", row.names = F)

# Prep validation tweets
df_val <- fread("./data/raw/user_val_tweets.csv")

# Average hand-code together to get final score:
df_val[, score := (scorer_1 + scorer_2)/2]

df_val[, text := gsub("http\\S+", "", text)]
df_val[, text := gsub("/t.co\\S+", "", text)]
df_val[, text := gsub("<.*>", "", text)]
df_val[, text := gsub("[^\x01-\x7F]", "", text)]

df_val[, author_id := as.numeric(as.factor(username))]
df_val[, id := .I]

write.csv(df_val, "./data/processed/user_val_tweets_processed.csv", row.names = F)

