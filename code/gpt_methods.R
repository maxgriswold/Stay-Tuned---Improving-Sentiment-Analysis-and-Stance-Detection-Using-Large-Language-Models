# Apply GPT to unstructured text to estimate sentiment (and stance)
# Max Griswold
# 5/2/2023

set.seed(2985671)

library(httr)
library(jsonlite)
library(data.table)
library(plyr)
library(dplyr)
library(R.utils)
library(parallel)
library(pbmcapply)

args <- commandArgs(trailingOnly = T)

# Since we can run multiple requests simultaneously for tuned 
# models but not for pretrained, use bash submision script to
# launch multiple instances to get tuned model results. Otherwise,
# use sequential requests to a given API.

if (length(args) > 1){
  tune <- T
  specific_model  <- args[1]
  specific_subject <- args[2]
  print("Running tuned GPT models")
}else{
  print("Running pretrained GPT models")
  specific_model <- args[1]
  tune <- F
}

# If recovery == F, get initial set of results. recovery == T attempts to check each file
# to see if any estimates were lost due to hitting internal API rate limits or timeouts, 
# adding these results to existing files. See log files to determine if recovery might be 
# needed.

recovery <- F

if (recovery){
	if (!dir.exists("./data/results/recovery")){
		dir.create("./data/results/recovery")
	}
}

# Load datasets
df_pol  <- fread("./data/pol_tweets_processed.csv")
df_user <- fread("./data/user_validate_tweets_processed.csv")
df_li   <- fread("./data/li_tweets_processed.csv")
df_kaw  <- fread("./data/kawintiranon_tweets_processed.csv")

# Load API information
api_info <- fread("./data/supplement/api_info.csv")

# Set up arguments for different model versions:

# Subjects: Biden 
#           Trump
# Dataset: Politician, 
#          User, 
#          Li 2021, 
#          Kawintiranon 2021
# Models: GPT-3.5 Turbo (Tuned - Party, Nominate, Hand), 
#         GPT-4, 
#         GPT-4o (Tuned - Party, Nominate, Hand), 
# Prompts: 3 different zero-shot prompts, 
#          Binary prompt, 
#          4-shot prompt, 
#          Chain-of-reasoning prompt

subjects <- c("trump", "biden")

analysis_datasets <- list("pol" = df_pol, 
                         "user_val" = df_user,
                         "li_2021" = df_li,
                         "kawintiranon_2021" = df_kaw)

# Set up models. Using two separate vectors to ensure only
# specific prompts are associated with specific model versions.
llm_pre <- c("gpt35", "gpt4", "gpt4o")
llm_tune <- c("gpt35_tune_party", "gpt35_tune_nominate", "gpt35_tune_handcode",
              "gpt4o_tune_party", "gpt4o_tune_nominate",  "gpt4o_tune_handcode")

# Set up prompts:
prompt_1 <- "Provide a continuous-value score between [-1, 1] that determines text's sentiment concerning the primary subject. Put the response in the following format, replacing the curly brackets with the score value. Score:{score}"
prompt_2 <- "Provide a continuous-value score between [-1, 1] that determines the text's sentiment directed towards {person}. Put the response in the following format, replacing the curly brackets with the score value. Score:{score}"          
prompt_3 <- "Provide a continuous-value score between [-1, 1] that determines whether the author of the author of the text likes or dislikes {person}, with -1 indicating greatest possible dislike and 1 indicating greatest possible like. Put the response in the following format, replacing the curly brackets with the score value. Score:{score}"

# Prompt 4 was adapted  for tuning models. When model is tuned, used 4a instead of 4.

if (tune){
  prompt_4 <- "Provide a score between [-1, 1] that determines whether the author of the text likes or dislikes {person}, with -1 indicating the greatest dislike and 1 indicating the greatest like. Put the response in the following format, replacing the curly brackets with the score value. Score:{score}"
}else{
  prompt_4 <- "Provide a binary-value score that is either -1 or 1 that determines whether the author of the text likes or dislikes {person}, with -1 indicating dislike and 1 indicating like. Put the response in the following format, replacing the curly brackets with the score value. Score:{score}"
}

prompt_5 <- "few_shot"

# Chain-of-Reasoning prompt
prompt_6 <- "chain_prompt"

prompt_list <- list("p1" = prompt_1,
                    "p2" = prompt_2,
                    "p3" = prompt_3,
                    "p4" = prompt_4,
                    "p5" = prompt_5,
                    "p6" = prompt_6)

# Build out analyses to run.
analysis_models <- setDT(expand.grid("llm_mod" = c(llm_pre, llm_tune),
                                     "subject_name" = subjects,
                                     "prompt_name" = names(prompt_list),
                                     "df_name" = names(analysis_datasets)))

# Since models are tuned separately by subject, we need to run tuned version by model/subject,
# while pretrained models can run for both subjects.
if (tune == T){
  analysis_models <- analysis_models[llm_mod == specific_model & subject_name == specific_subject & prompt_name == "p4",]
}else{
  analysis_models <- analysis_models[llm_mod == specific_model,]
}

# Crete information on dataset sizes to set how many independent 
# API requests we will need to make:
data_size <- expand.grid("subject_name" = subjects,
                         "df_name" = names(analysis_datasets))

# Below is prespecified based on the number of rows in each dataset filtered by subject
data_size$n <- c(14934, 5508, 376, 377, 6362, 5806, 1250, 1250)

analysis_models <- join(analysis_models, data_size, type = "left")

run_llm_analysis <- function(llm_mod, subject_name, prompt_name, df_name, temp = 0){
  
  # Loop over tweets of chosen dataset:
  dd <- analysis_datasets[[df_name]]
  dd <- dd[subject == subject_name,]
  
  # We are rate limited to 500 requests per minute (and token limited).
  # So, in batches, get scores and write to disk, then wait for remaining minute. 
  # I'm using slightly less requests than provided to ensure we are able to get all
  # responses. At full RPM, we sometimes lose responses. 

  # We need to submit multiple requests in a row for this prompt.
  # So, reduce batch size according to number of sequential requests (3):

  if (prompt_name == "p6"){
    # For this prompt, GPT sometimes produces a ton of extra text.
    # and we need to run 3 prompts in succession.
    # So, run in even smaller batches to ensure we don't hit the token limit
    batch_size <- 99/3

    # Specify number of parallel jobs
    core_num <- 16
    
  }else{
    batch_size <- 100
    core_num <- 16
  }

  # For tuned models, need lower RPM 
  if (tune){
    batch_size <- 100
    core_num <- 4
  }

  batches <- c(seq(1, dim(dd)[1], batch_size), dim(dd)[1])
  print(sprintf("Estimating model %s %s %s %s", llm_mod, df_name, prompt_name, subject_name))
  
  for (i in 1:(length(batches) - 1)){

    begin_time <- Sys.time()

    print(sprintf("Starting batch %s of %s", i, length(batches) - 1))
    
    segment <- dd[batches[i]:batches[i + 1]]$id
    
    # The purpose of the odd-looking code below is to ensure the parallel process
    # job completes in a given timeframe. So, we're passing a given tweet_id within
    # a series of ids to the primary function we're  seeking to execute - query_api - and 
    # wrapping it with timeout call. 
    completion <- pbmclapply(segment, function(x) timeout_wrapper(query_api, list(tweet_id = x, dd = dd, 
                           df_name = df_name, prompt_name = prompt_name, 
                           llm_mod = llm_mod, subject_name = subject_name), timeout = 120), 
                           mc.cores = core_num, mc.style = "ETA")                   

    # completion <- lapply(segment, function(x) {
    #                     timeout_wrapper(query_api, args = list(tweet_id = x,
    #                     dd = dd, df_name = df_name, prompt_name = prompt_name, 
    #                     llm_mod = llm_mod, subject_name = subject_name), timeout = 180)})

    # Some cores have been returning errors so this function tests is the
    # return from GPT is an expected dataframe. If not, save the model
    # information and text id to rerun results later. This all captures instances
    # when parallelization has a timeout.

    # What are the indices for list items which are not dataframes? 
    missing_results <- which(!sapply(completion, is.data.frame))

    if (length(missing_results) > 0){

      print(sprintf("Lost estimates for %s tweets", length(missing_results)))

      lost_estimates <- data.table("llm_mod" = llm_mod, "df_name" = df_name, "prompt_name" = prompt_name,
                                   "subject_name" = subject_name, "tweet_id" = segment[missing_results])
                                   
      filename_lost_estimates <- sprintf("./data/results/lost_estimates_gpt35.csv")
        
      if (!file.exists(filename_lost_estimates)){
        write.table(lost_estimates, filename_lost_estimates, sep = ",", row.names = F)
      }else{
        write.table(lost_estimates, filename_lost_estimates, sep = ",", append = T, row.names = F, col.names = F)
      }
      
      print(completion[missing_results])

      completion <- completion[-missing_results]
  
    }
    
    # Check to make sure all results are not missing, based on previous test:
    # Check to make sure all results are not missing, based on previous test:
    if (length(completion) > 0){

      results <- rbindlist(completion)
      
      percent_complete <- paste0(round((i)/(length(batches) - 1), 3)*100, "%")
      print(sprintf("Finished %s. Writing to disk!", percent_complete))
      
      # Save results to disk. This is highly inefficient but we are rate-limited
      # anyway and this ensures we do not lose a batch if an error occurs due to a
      # network disconnect.
      
      filename <- sprintf("./data/results/%s_%s_%s_%s.csv", llm_mod, df_name, prompt_name, subject_name)
      
      if (batches[i] == 1){
        if (file.exists(filename)){
          file.remove(filename)
        }
        file.create(filename)
        write.table(results, file = filename, sep = ",", row.names = F)
      }else{
        write.table(results, file = filename, append = T, sep = ",", row.names = F, col.names = F)
      }
    }else{
      print("Lost batch")
    }

    #heck timing: nudge to one minute and five seconds to be safe:
    time_diff <- as.numeric(65, units = "secs") - as.numeric(Sys.time() - begin_time, units = "secs")
    if (time_diff > 0 & time_diff < 65){
      Sys.sleep(time_diff)
    }
  }  

  return()

}

query_api <- function(tweet_id, dd, df_name, prompt_name, 
                      llm_mod, subject_name, temp = 0){
  
  tweet   <- dd[id == tweet_id]$text
  
  # Based on the LLM model and subject, use a different api:
  api_version <- api_info[short_name == llm_mod & subject == subject_name,]
  
  # Prompts 1-4 use a similar format.
  # For prompt 5, we need to randomize the few-shots,
  # following Zhao, 2024 suggestions.. 
  # For prompt 6, we need to have a series of messages with
  # GPT to enable chain-of-though, as per Zhang, 2024

  if (prompt_name %in% paste0("p", 1:4)){
    
    # Add on specific tweet to code and replace subject-hash.
    p <- paste0(prompt_list[[prompt_name]], "\n'", tweet, "'")
    p <- gsub("\\{person\\}", subject_name, p)
    
    res <- POST(url = api_version$url,
                add_headers("Ocp-Apim-Subscription-Key" = api_version$key),
                content_type_json(),
                encode = 'json',
                body = list(
                  model = api_version$model,
                  messages = list(list(role = 'user', content = p)),
                  temperature = temp
                ))
    
    res_score <- httr::content(res)$choices[[1]]$message$content
  
  }
  
  if (prompt_name == "p5"){
    
    # For politician dataset, use nominate as score for example:
    # Make sure to reverse code the index, if subject is biden,
    # so that liberals are coded as positive and repubs as negative.
    if (subject_name == "biden" & df_name == "pol"){
      dd[, score := nominate_dim1*-1]
    }else if (subject_name == "trump" & df_name == "pol"){
      dd[, score := nominate_dim1]
    }
    
    # Few-shot prompt, following Brown et al., 2020, Li and Conrad, 2024, and Zhao et al., 2021
    # Based on Zhao et al., 2021, it is best if we randomize the ordering
    # of few shot examples.
    
    example_tweets <- sample(dd[id != tweet_id,]$id, 4)
    examples <- ""
    
    for (i in example_tweets){
      
      t <- dd[id == i]$text
      s <- dd[id == i]$score
      
      examples <- paste0(examples, "text: ", t, "\nScore:", s, "\n")
      
    }
    
    p <- paste0(prompt_3, "\n", examples, "\n", "text: ", tweet, "\nScore:")
    p <- gsub("\\{person\\}", subject_name, p)
    
    res <- POST(url = api_version$url,
                add_headers("Ocp-Apim-Subscription-Key" = api_version$key),
                content_type_json(),
                encode = 'json',
                body = list(
                  model = api_version$model,
                  messages = list(list(role = 'user', content = p)),
                  temperature = temp
                )) 
    
    res_score <- httr::content(res)$choices[[1]]$message$content
    
  }
  
  if (prompt_name == "p6"){
    
    # Use a random text besides the one we are estimating to construct
    # the chain-of-thought example.
    
    other_tweet <- sample(dd[id != tweet_id]$text, 1)
    base_prompt <- paste0(prompt_3, "\n'", other_tweet, "'")
    base_prompt <- gsub("\\{person\\}", subject_name, base_prompt)
    
    res_base <- POST(url = api_version$url,
                add_headers("Ocp-Apim-Subscription-Key" = api_version$key),
                content_type_json(),
                encode = 'json',
                body = list(
                  model = api_version$model,
                  messages = list(list(role = 'user', content = base_prompt)),
                  temperature = temp
                ))
    
    res_base <- httr::content(res_base)$choices[[1]]$message$content
    
    # Get context for response from GPT:
    context_prompt <- "Why?"
    
    res_context <- POST(url = api_version$url,
                     add_headers("Ocp-Apim-Subscription-Key" = api_version$key),
                     content_type_json(),
                     encode = 'json',
                     body = list(
                       model = api_version$model,
                       messages = list(list(role = 'user', content = base_prompt),
                                       list(role = 'assistant', content = res_base),
                                       list(role = 'user', content = context_prompt)),
                       temperature = temp
                     )) 
    
    res_context <- httr::content(res_context)$choices[[1]]$message$content
    
    # Add context as example prior to target text:
    add_prompt <- paste0(prompt_3, "\n'", tweet, "'")
    add_prompt <- gsub("\\{person\\}", subject_name, add_prompt)
    
    target_prompt <- paste0("Q: ", base_prompt, "\n A: Let's think this through step-by-step. ",
                            res_context, "\n", res_base, "\n Q: ", add_prompt)
    
    res <- POST(url = api_version$url,
                        add_headers("Ocp-Apim-Subscription-Key" = api_version$key),
                        content_type_json(),
                        encode = 'json',
                        body = list(
                          model = api_version$model,
                          messages = list(list(role = 'user', content = target_prompt)),
                          temperature = temp
                        )) 
    
    res_score <- httr::content(res)$choices[[1]]$message$content
    
  }

  sink(sprintf("./outputs/%s_%s_%s.txt", llm_mod, df_name, subject_name), append = T)
  print(tweet_id)
  print(res)
  print(res_score)
  sink()
  
  # Extract score from the response, if possible:
  score <- as.numeric(gsub(".*Score:|\\{|\\}", "", res_score))
  
  df_res <- data.table("id" = tweet_id, "sentiment_tweet" = score)
  if (!is.data.frame(df_res)|is.null(res_score)){
    df_res <- "Error: See log file"
  }

  return(df_res)
  
}

# Wrapper to ensure the query function does not stay hanging when parallelized, 
# causing jobs to stop. This could occur if the API takes too long to send back a
# request. If we don't receive a request within 3 minutes, cancel the job:

timeout_wrapper <- function(func, args = list(), timeout = 180){

  stopifnot(is.function(func))

  result <- tryCatch({
    withTimeout({
      do.call(func, args)
    }, timeout = timeout)
  }, TimeoutException = function(e){
    "Timed out"
  }, error = function(e) {
      paste("Error: ", e$message)
  })

  return(result)

}

recover_results <- function(llm_mod, subject_name, prompt_name, df_name){

  # Load existing estimate file and compare results against the underlying
  # text data. Determine which row ids are missing due to hitting request limits
  # or timeout:

  dd <- analysis_datasets[[df_name]]
  dd <- dd[subject == subject_name,]
  
  filename <- sprintf("./data/results/%s_%s_%s_%s.csv", llm_mod, df_name, prompt_name, subject_name)
  dd_est <- fread(filename)

  # Ensure column names were set (some files were missing colnames):
  setnames(dd_est, names(dd_est), c("id", "sentiment_tweet"))

  # Ensure we didn't add any duplicates into a file:
  dd_est <- unique(dd_est)
  dd_est <- dd_est[, .SD[1], by = "id"]

  # Also try getting estimates for rows that were NA due to responses
  # from GPT that did not align with expected pattern:
  dd_est <- dd_est[!is.na(sentiment_tweet)]

  est_id <- dd_est$id
  missing_id <- dd[!(id %in% est_id)]$id

  if (length(missing_id) > 0){
    if (prompt_name == "p6"){
      # For this prompt, GPT sometimes produces a ton of extra text.
      # and we need to run 3 prompts in succession.
      # So, run in even smaller batches to ensure we don't hit the token limit
      batch_size <- 199/3

      # Specify number of parallel jobs
      core_num <- 16
      
    }else{
      batch_size <- 200
      core_num <- 16
    }

    # For tuned models, need lower RPM 
    if (tune){
      batch_size <- 150
      core_num <- 4
    }

    batches <- c(seq(1, length(missing_id), batch_size), length(missing_id))
    print(sprintf("Recovering estimates for model %s %s %s %s", llm_mod, df_name, prompt_name, subject_name))
    print(sprintf("Model was missing %s estimates", length(missing_id)))

    for (i in 1:(length(batches) - 1)){

      begin_time <- Sys.time()

      print(sprintf("Starting batch %s of %s", i, length(batches) - 1))
      
      segment <- missing_id[batches[i]:batches[i + 1]]
      
      # Providing a little extra time for recovery requests and ommitting the progress bar:
      completion <- mclapply(segment, function(x) timeout_wrapper(query_api, list(tweet_id = x, dd = dd, 
                            df_name = df_name, prompt_name = prompt_name, 
                            llm_mod = llm_mod, subject_name = subject_name), timeout = 180), 
                            mc.cores = core_num)

      missing_results <- which(!sapply(completion, is.data.frame))
      if (length(missing_results) > 0){
        completion <- completion[-missing_results]
      }

      if (length(completion) == length(segment)){
        print("All estimates recovered in missing batch!")
        completion <- rbindlist(completion)
        dd_est <- rbindlist(list(dd_est, completion))                   
      }else if (length(completion) > 0){
        print(sprintf("Recovered %s of %s in missing batch", length(completion), length(segment)))
      }else{
        print(sprintf("Did not recover %s missing batch", length(segment)))
      }

      #Check timing: nudge to one minute and five seconds to be safe:
      time_diff <- as.numeric(65, units = "secs") - as.numeric(Sys.time() - begin_time, units = "secs")
      if (time_diff > 0 & time_diff < 65){
        Sys.sleep(time_diff)
      }
    }

    # Save updated file:
    old <- length(missing_id)

    new_ids <- dd_est$id
    still_missing <- dd[!(id %in% new_ids)]$id
    new <- old - length(still_missing)

    # Ensure we didn't add any duplicates into a file:
    dd_est <- unique(dd_est)
    dd_est <- dd_est[, .SD[1], by = "id"]

    print(sprintf("Saving. Added %s of %s estimates missing from results", new, old))

    filename <- sprintf("./data/results/recovery/%s_%s_%s_%s.csv", llm_mod, df_name, prompt_name, subject_name)
    write.table(dd_est, file = filename, sep = ",", row.names = F)

  }else{

    print(sprintf("No missing estimates!"))
  
    filename <- sprintf("./data/results/recovery/%s_%s_%s_%s.csv", run_date, llm_mod, df_name, prompt_name, subject_name)
    write.table(dd_est, file = filename, sep = ",", row.names = F)

  }

  # In addition to log file, save dataset detailing N by model to review when validation finishes:
  df_recovery <- data.table("llm_mod" = llm_mod, "df_name" = df_name, "prompt_name" = prompt_name,
                            "subject_name" = subject_name, "n_ests" = dim(dd_est)[1], "n_data" = dim(dd)[1])

  filename_recovery <- sprintf("./data/results/estimate_recovery_tracking.csv", run_date)

  if (!file.exists(filename_recovery)){
    write.table(df_recovery, filename_recovery, sep = ",", row.names = F)
  }else{
    write.table(df_recovery, filename_recovery, sep = ",", append = T, row.names = F, col.names = F)
  }

  return()

}

# Run models sequentially (given rate limits; otherwise I would suggest parallelizing runs)

# Run the models
if (recovery == F){
  for (i in 1:dim(analysis_models)[1]){
    run_llm_analysis(analysis_models[i]$llm_mod, 
                    analysis_models[i]$subject_name,
                    analysis_models[i]$prompt_name,
                    analysis_models[i]$df_name)
  }
}else{
  for (i in 1:dim(analysis_models)[1]){
    recover_results(analysis_models[i]$llm_mod, 
                      analysis_models[i]$subject_name,
                      analysis_models[i]$prompt_name,
                      analysis_models[i]$df_name)
  }
  print("Finished all models")
}

  