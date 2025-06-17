# Sentiment method plots
# Max Griswold
# 6/21/23

rm(list = ls())

library(ggplot2)
library(irr)

# Extensions to ggplot for various bespoke needs
library(ggridges)  # For geom_ridges
library(gridExtra) # For arrangeGrob, to pass ggplots from function
library(grid)      # For grid.draw, after collecting a list of grobs
library(cowplot)   # For plot_grid, arranging p lots in a specific orientation
library(ggforce)   # For facet_wrap_paginate
library(ggpubr)    # Easy arrange w/ separate legend
library(ggstance)  # For position_dodgev

library(extrafont)
font_import()
loadfonts(device="win")

library(data.table)

library(plyr)
library(dplyr)
library(stringr)
library(readxl)

library(kableExtra)      # For generating Latex tables from dataframes

df <- fread("./data/results/analysis_results.csv")

outcome <- c("cont", "bin", "cat")

data_names      <- c('pol', 'user', 'kawintiranon', 'li')
data_names_long <- c("116th U.S. Congress", "Twitter Users", 
                     "Kawintiranon & Singh, 2021",  "P-Stance")

data_name_lookup <- data.table("data_name" = data_names, "data_name_long" = data_names_long)

# Rename variables in the plot; note, position order matters
model_name_full <- c("GPT-4 Omni", "GPT-4", "GPT-3.5 Turbo", "DeBERTa-NLI", "DistilBERT Uncased",
                      "SiEBERT", "RoBERTa-TweetEval", "VADER", "Hu & Liu", "NRC", "SenticNet 4",
                      "SO-CAL", "SentiWordNet")

model_name_short <- c("gpt4o", "gpt4", "gpt35", "deberta", "distilbert", "siebert", 
                       "tweetnlp", "vader", "huliu", "nrc", "senticnet", "socal_google", "sentiword")     

method_type <- c(rep("Large \nLanguage \nModel", 3), rep("Supervised \nLanguage \n Model", 4), rep("Lexical", 6))

method_lookup <- data.table("model_name" = model_name_short, "model_name_long" = model_name_full, "method_type" = method_type)

df <- join(df, method_lookup, type = 'left', by = 'model_name')
df <- join(df, data_name_lookup, type = 'left', by = 'data_name')

# Reorder methods for plotting purposes
df[, model_name := factor(model_name, levels = rev(model_name_short))]
df[, model_name_long := factor(model_name_long, levels = rev(model_name_full))]

# For reordering, I'm adding an additional omitted category group. This group will
# correspond to validation data added to the dataframe when plotting the empirical distributions.
df[, method_type := factor(method_type, levels = c("Large \nLanguage \nModel", "Supervised \nLanguage \n Model", 
                                                   "Lexical", "Validation"))]

# Change prompt, tune, and subject names to make plots look a little nicer
df[, prompt_name_long := mapvalues(prompt_name, paste0("p", 1:6),
                                   c("Sentiment", "Stance (Alt)", "Stance", 
                                     "Stance (Binary)",  "Few-Shot", "Chain-of-Thought"))]

df[, prompt_name_long := factor(prompt_name_long, levels = c("Sentiment", "Stance (Alt)", "Stance (Binary)", 
                                                             "Stance", "Few-Shot", "Chain-of-Thought"))]

df[, subject := str_to_title(subject)]
df[, tune_name_long := mapvalues(tune_data, c("handcode", "party", "nominate"),
                                  c("Tuned (Human-Coded)", "Tuned (Party Affiliation)", "Tuned (DW-Nominate)"))]

df[, tune_name_long := factor(tune_name_long, levels = c("Tuned (Human-Coded)", "Tuned (Party Affiliation)", "Tuned (DW-Nominate)"))]

#################
# Example Texts #
#################

pol_text  <- fread("./data/processed/pol_tweets_processed.csv")[, .(id, text)]
user_text <- fread("./data/processed/user_train_tweets_processed.csv")[, .(id, text, subject)]

pol_text <- pol_text[id %in% c(789, 2617, 16836, 10910),]
user_text <- user_text[id %in% c(289, 752, 375, 2), ]

df_pol <- join(pol_text, df[data_name == 'pol'], by = 'id', type = 'left')[, .(id, text, data_name, subject, model_name_long, tune_name_long, prompt_name_long, est_score)]
df_user <- join(user_text, df[data_name == 'user'], by = 'id', type = 'left')[, .(id, text, data_name, subject, model_name_long, tune_name_long, prompt_name_long, est_score)]

df_text <- as.data.table(rbind(df_pol, df_user))

# For each text, display 3 side-by-side tables of results for pretrained, tuned, and prompts:
df_text[,text_type := ifelse(is.na(tune_name_long) & !is.na(prompt_name_long), "LLM Prompts",
                              ifelse(!is.na(tune_name_long), "Tuned Models", "Pretrained Models"))]

unique_tweets <- unique(df_text$text)

# Initialize LaTeX output
output <- ""

for (tweet in unique_tweets) {
  
  formatted_tweet <- str_replace_all(tweet, "([#&])", "\\\\\\1")
  formatted_tweet <- str_replace_all(formatted_tweet, "([#@]\\S+)", "\\\\texttt{\\1}")
  output <- paste0(output, "\\begin{quote}\n", formatted_tweet, "\n\\end{quote}\n")
  
  # Filter the dataframe for the current text
  dd <- df_text[text == tweet,]
  dd[, est_score := round(est_score, 3)]
  
  # Create a table for each text_type
  tables <- list()
  
  for (type in c("Pretrained Models", "Tuned Models", "LLM Prompts")){
    dd_sub <- dd[text_type == type, ]
    
    dd_sub[model_name_long == "Hu & Liu", model_name_long := "Hu \\& Liu"]
    
    if (type == "Pretrained Models"){
      dd_sub <- dd_sub[, .(model_name_long, est_score)]
      col_names <- c("Model", "Score")
    }else if (type == "Tuned Models"){
      dd_sub <- dd_sub[, .(model_name_long, tune_name_long, est_score)]
      dd_sub[, tune_name_long := gsub("Tuned|(|)", "", tune_name_long)]
      col_names <- c("Model", "Tune Data", "Score")
    }else{
      dd_sub <- dd_sub[, .(model_name_long, prompt_name_long, est_score)]
      col_names <- c("Model", "Prompt", "Score")
    }
    
    tables[type] <- kable(dd_sub, format = "latex", booktabs = T, escape = F, 
                          col.names = col_names, caption = type) %>%
                    kable_styling(latex_options = c("striped", "scale_down", "HOLD_position")) %>%
                    gsub("\\\\addlinespace", "", .)
  }   
  
  # Combine tables side by side
  combined_tables <- paste(tables, collapse = "\n\n\\vspace{0.5cm}\n\n")
  
  # Append combined tables to LaTeX output
  output <- paste0(output, combined_tables, "\n\\hrulefill\n\n")
  
}

# Write the LaTeX output to a file
writeLines(output, "./paper/latex/example_texts.tex")

#########################################
# Distribution Plots (EDF v. Estimates) #
#########################################

pol_val <- fread("./data/raw/supplement/politician_tweets_handcoded.csv")[, .(id, score, subject)]
pol_id  <- fread("./data/processed/pol_tweets_processed.csv")[, .(id, party_code, nominate_dim1, subject)]

plot_dist <- function(d_name, dd, include_tuned = F){
  
  dd <- dd[data_name == d_name, .(model_name_long, method_type, prompt_name_long, tuned, score,
                                         tune_name_long, subject, data_name_long, both_subjects, id, est_score)]

  # For politician dataset, get the sample we handcoded, along with
  # party identification for each tweet ID. Make plots by party identification
  # and include multiple validation sets (nominate dimension one & coded scores) 
  
  if (d_name == 'pol'){
    
    dd_add <- join(pol_id, pol_val, by = c("id", "subject"))
    
    dd[, score := NULL]
    
    # Add binary score based on party_code and subject.
    # If the target is Biden, then Dem = 1 and Repub = -1,
    # Otherwise, if target is Trump, Dem = -1 and Repub = 1
    dd_add[, `Score (Party)` := ifelse(subject == "biden", 
                                      ifelse(party_code == "D", 1, -1),
                                      ifelse(party_code == "D", -1, 1))]
    
    # Reverse-code nominate scores when the target is Biden. Nominate dimension 1
    # indicates party affiliation from left to right (-1 to 1). So when the target
    # is Biden, we need a reverse coding to align with the other score metrics:
    dd_add[, `Score (DW-Nominate)` := ifelse(subject == "biden", nominate_dim1*-1, nominate_dim1)]
    
    setnames(dd_add, "score", "Score (Human-Coded)")
    
    dd_party <- unique(dd_add[, .(party_code, id)])
    df_plot <- join(dd, dd_party, by = c("id"), type = 'left')
    
  }else if(d_name == 'user'){
    
    # Add on score as validation data for comparing distributions
    dd_validate <- unique(dd[, .(id, subject, both_subjects, score)])
    dd_validate <- melt(dd_validate, id.vars = c("id", "subject", "both_subjects"), variable.name = "model_name_long", value.name = "est_score")
    dd_validate[, `:=`(data_name_long = unique(dd$data_name_long), 
                       tune_name_long = NA, prompt_name_long = NA, tuned = F,
                       subject = str_to_title(subject),
                       method_type = "Validation", model_name_long = "Score (Human-Coded)")]
    
    dd[, score := NULL]
    
    df_plot <- rbind(dd, dd_validate)

  }else{
    
    # Add on score as validation data for comparing distributions
    dd_validate <- unique(dd[, .(id, subject, score)])
    dd_validate <- melt(dd_validate, id.vars = c("id", "subject"), variable.name = "model_name_long", value.name = "est_score")
    dd_validate[, `:=`(data_name_long = unique(dd$data_name_long), 
                       tune_name_long = NA, prompt_name_long = NA, tuned = F,
                       subject = str_to_title(subject),
                       method_type = "Validation", model_name_long = "Score (Human-Coded)")]
    
    dd[, score := NULL]
    dd[, both_subjects := NULL]
    
    df_plot <- rbind(dd, dd_validate)
    
  }
  
  # Add additional detail to model names, including prompt_name if untuned and data used for tuning.
  df_plot[!is.na(prompt_name_long) & is.na(tune_name_long), model_name_long := paste0(model_name_long, ": ", prompt_name_long)]
  df_plot[!is.na(tune_name_long), model_name_long := paste0(model_name_long, ": ", tune_name_long)]
  
  # Display tuned data?
  if (include_tuned){
    df_plot <- df_plot[(tuned == T|(method_type == "Validation")),]
  }else{
    df_plot <- df_plot[tuned == F & prompt_name_long %in% c("Stance", NA)]
  }
  
  if (include_tuned){
    
    # Change method types to make each category of models clearer:
    df_plot[model_name_long %like% "GPT-4 Omni", method_type := "GPT-4 Omni"]
    df_plot[model_name_long %like% "GPT-4" & method_type == "LLM", method_type := "GPT-4"]
    df_plot[model_name_long %like% "GPT-3.5", method_type := "GPT-3.5 Turbo"]
    
    # Change GPT model names to simply be the prompt:
    df_plot[, model_name_long := gsub("GPT.*: ", "", model_name_long)]
    
    df_plot[model_name_long %like% "SiEBERT", method_type := "SiEBERT"]
    df_plot[model_name_long %like% "RoBERTa-TweetEval", method_type := "RoBERTa-TweetEval"]
    df_plot[model_name_long %like% "DistilBERT Uncased", method_type := "DistilBERT Uncased"]
    df_plot[model_name_long %like% "DeBERTa-NLI", method_type := "DeBERTa-NLI"]
    
    # Change transfer learning names to be the tune set:
    df_plot[, model_name_long := gsub("SiEBERT: |DeBERTa-NLI: |DistilBERT Uncased: |RoBERTa\\-TweetEval: ", "", model_name_long)]
    
    df_plot[, method_type := factor(df_plot$method_type, 
                                    levels = c("GPT-4 Omni", "GPT-4", "GPT-3.5 Turbo", "DeBERTa-NLI", "SiEBERT", "DistilBERT Uncased", "RoBERTa-TweetEval", "Validation"))]
    
  }else{
    
    # If pretrained, remove prompt_name from model_name
    df_plot[, model_name_long := gsub(": P3", "", model_name_long)]
    
  }
  
  df_plot[, model_name_long := as.character(model_name_long)]
  
  p <- ggplot(df_plot, aes(y = model_name_long)) +
          scale_y_discrete(expand = expansion(add = c(0.3, 0.3))) +
          scale_x_continuous(expand = expansion(add = c(0.3, 0.3)),
                             breaks = c(-1, 0, 1),
                             limits = c(-1.3, 1.3)) +
          labs(x = "Stance Score", 
               y = NULL,
               title = "") +
          facet_grid(method_type ~ subject, space = 'free_y', scale = 'free_y', switch = 'y') +
          coord_cartesian(clip = "off") +
          theme_ridges(grid = T) +
          theme(strip.background = element_blank(),
                legend.position = "bottom",
                axis.title.y = element_text(angle = 0, vjust = 1),
                plot.title = element_text(hjust = 0),
                strip.text = element_text(size = 12, family = 'sans'),
                strip.text.y.left = element_text(angle = 0, hjust = 0, margin = margin(r = 10)), 
                axis.text.y = element_text(hjust = 1),
                strip.placement = 'outside',
                panel.spacing = unit(1, "lines"),
                plot.subtitle = element_text(hjust = 0),
                axis.text = element_text(size = 12, family = "sans"),
                axis.title = element_text(size = 14, family = "sans"))
  
  if (d_name == 'pol'){
    p <- p + 
          geom_density_ridges(alpha = 0.5, 
                              aes(x = est_score, fill = party_code),
                              scale = 1.1, rel_min_height = 0.03) + 
          # scale = 0.9, stat = 'binline', bins = 30) +
          scale_fill_manual(breaks = c("R", "D"),
                            labels = c("Republican", "Democrat"),
                            values = c("#ff8080", "#8080ff"),
                            name = "Party", guide = 'legend') 
    
  }else if (d_name == 'user'){
    p <- p +
      geom_density_ridges(alpha = 0.5, 
                          aes(x = est_score, fill = both_subjects),
                          scale = 1.1) +
      scale_fill_manual(breaks = c(T, F),
                        labels = c("Yes", "No"),
                        values = c("#1f78b4", "#33a02c"),
                        name = "Both subjects mentioned?", guide = 'none') 
      
   p_a <- p %+% df_plot[both_subjects == T, ] + labs(subtitle = "Single subject", x = "") + guides(fill = 'none')
   p_b <- p %+% df_plot[both_subjects == F, ] + labs(title = "", subtitle = "Multiple subjects")
   
   p <- plot_grid(p_a, p_b, align = 'v', nrow = 2)
    
  }else{
    p <- p +
      geom_density_ridges(alpha = 0.5, 
                          aes(x = est_score),
                          scale = 0.9)
  }
  
  return(p)
  
}

# For main paper, only plot results for selected samples:
select_models   <- c("gpt4o", "gpt35", "deberta", "siebert")
select_prompts   <- c("p3", NA)
select_tune_data <- c("nominate", "handcode")

df_select <- df[(model_name %in% select_models & prompt_name %in% select_prompts & tuned == F)]

dist_select <- lapply(data_names, plot_dist, dd = df_select, include_tuned = F)

ggsave("./figs/Fig 1.pdf", 
       plot = dist_select[[1]], 
       width = 11.69, 
       height = 8.27)

cairo_ps("./figs/Fig 1.eps", height = 8.27, width = 11.69)
dist_select[[1]]
dev.off()

ggsave("./figs/Fig 1.eps", 
       plot = dist_select[[1]], 
       device = cairo_ps,
       width = 11.69, 
       height = 8.27,
       units = "in",
       dpi = 72)

ggsave("./figs/SA1 Fig 1.pdf", 
       plot = dist_select[[2]], 
       width = 11.69, 
       height = 8.27)

cairo_ps("./figs/SA1 Fig 1.eps", height = 8.27, width = 11.69)
dist_select[[2]]
dev.off()

ggsave("./figs/SA1 Fig 2.pdf", 
       plot = dist_select[[3]], 
       width = 11.69,
       height = 8.27)

cairo_ps("./figs/SA1 Fig 2.eps", height = 8.27, width = 11.69)
dist_select[[3]]
dev.off()

ggsave("./figs/SA1 Fig 3.pdf", 
       plot = dist_select[[4]], 
       width = 11.69, 
       height = 8.27)

cairo_ps("./figs/SA1 Fig 3.eps", height = 8.27, width = 11.69)
dist_select[[4]]
dev.off()

select_models   <- c("gpt4o", "gpt35", "deberta", "siebert")
select_tune_data <- c("nominate", "handcode")

df_select <- df[(model_name %in% select_models) & ((prompt_name %in% select_prompts) | (tuned == T))]

dist_select <- lapply(data_names, plot_dist, dd = df_select, include_tuned = T)

ggsave("./figs/SA1 Fig 4.pdf", 
       plot = dist_select[[1]], 
       width = 11.69*2, 
       height = 8.27*2)

cairo_ps("./figs/SA1 Fig 4.eps", height = 8.27, width = 11.69)
dist_select[[1]]
dev.off()

ggsave("./figs/SA1 Fig 5.pdf", 
       plot = dist_select[[2]], 
       width = 11.69*2, 
       height = 8.27*2)

cairo_ps("./figs/SA1 Fig 5.eps", height = 8.27, width = 11.69)
dist_select[[2]]
dev.off()

ggsave("./figs/SA1 Fig 6.pdf", 
       plot = dist_select[[3]], 
       width = 11.69*2, 
       height = 8.27*2)

cairo_ps("./figs/SA1 Fig 6.eps", height = 8.27, width = 11.69)
dist_select[[3]]
dev.off()

ggsave("./figs/SA1 Fig 7.pdf", 
       plot = dist_select[[4]], 
       width = 11.69*2, 
       height = 8.27*2)

cairo_ps("./figs/SA1 Fig 7.eps", height = 8.27, width = 11.69)
dist_select[[4]]
dev.off()


###########################
# Inter-rater reliability #
###########################

pol_scored <- fread("./data/raw/pol_tweets_scored.csv")[, .(text, subject, scorer_1, scorer_2)]

# Get the two sets of users scores: validation and train sets:
user_val   <- fread("./data/raw/user_val_tweets.csv")[, .(text, subject, scorer_1, scorer_2)]
user_train <- fread("./data/raw/user_train_tweets.csv")[, .(text, subject, scorer_1, scorer_2)]

df_combine <- rbindlist(list(pol_scored, user_train, user_val))

df_combine[, subject := ifelse(subject == "biden", "Biden", "Trump")]

rater_scatter <- ggplot(df_combine, aes(x = scorer_1, y = scorer_2)) +
                geom_abline(alpha = 1, linetype = 21, intercept = 0, slope = 1) +
                geom_smooth(method = 'lm', alpha = 0.5) +
                geom_point(alpha = 0.2) +
                facet_wrap(~subject, nrow = 2) +
                lims(x = c(-1, 1), y = c(-1, 1)) +
                theme_bw() +
                labs(x = "Annotator 1", 
                     y = "Annotator 2") +
                theme(strip.background = element_blank(),
                      legend.position = "bottom",
                      axis.title.y = element_text(angle = 0, vjust = 0.9),
                      plot.title = element_text(hjust = 0),
                      strip.text = element_text(size = 12, family = 'sans'),
                      strip.placement = 'outside',
                      panel.spacing = unit(1, "lines"),
                      plot.subtitle = element_text(hjust = 0),
                      axis.title = element_text(size = 12, family = "sans"))

ggsave("./figs/Fig S. Handcode Compare.pdf", rater_scatter,
       device = 'pdf', bg = "transparent",
       height = 8.27, width = 8.27, unit = 'in')

# Calculate overall ICC
icc(as.matrix(df_combine[, .(scorer_1, scorer_2)]), model = "oneway", unit = "single", type = "consistency")

# Convert to binary scores and calculate Cohen's Kappa:
df_combine[, scorer_1 := ifelse(scorer_1 > 0, 1, 0)]
df_combine[, scorer_2 := ifelse(scorer_2 > 0, 1, 0)]

kappa2(as.matrix(df_combine[, .(scorer_1, scorer_2)]), weight = "equal")

# Produce contingency table:
table(df_combine[subject == "Biden"]$scorer_1, df_combine[subject == "Biden"]$scorer_2)
table(df_combine[subject == "Trump"]$scorer_1, df_combine[subject == "Trump"]$scorer_2)

###########################################
# Scatterplots: ID, Nominate, Human-Coded #
###########################################

# Choosing any two model_id (biden+trump) to get one set of the validation data:
df_pol <- df[model_id %in% c(97, 98) & !is.na(score_coded), .(id, score, score_nominate, score_coded)]
df_pol <- join(df_pol, pol_id[, .(id, party_code, subject)], type = "left", by = "id")

df_pol[, subject := str_to_title(subject)]

# High correlation between coded score and these metrics. But
# are relative values well aligned within parties?

cor.test(df_pol$score, df_pol$score_coded)
cor.test(df_pol[subject == "Biden"]$score, df_pol[subject == "Biden"]$score_coded)
cor.test(df_pol[subject == "Trump"]$score, df_pol[subject == "Trump"]$score_coded)

cor.test(df_pol$score, df_pol$score_nominate)
cor.test(df_pol[subject == "Biden"]$score_nominate, df_pol[subject == "Biden"]$score_coded)
cor.test(df_pol[subject == "Trump"]$score_nominate, df_pol[subject == "Trump"]$score_coded)

cor.test(df_pol[party_code == "D"]$score, df_pol[party_code == "D"]$score_nominate)
cor.test(df_pol[subject == "Biden" & party_code == "D"]$score_nominate, df_pol[subject == "Biden" & party_code == "D"]$score_coded)
cor.test(df_pol[subject == "Trump"  & party_code == "D"]$score_nominate, df_pol[subject == "Trump"& party_code == "D"]$score_coded)

cor.test(df_pol[party_code == "R"]$score, df_pol[party_code == "R"]$score_nominate)
cor.test(df_pol[subject == "Biden" & party_code == "R"]$score_nominate, df_pol[subject == "Biden" & party_code == "R"]$score_coded)
cor.test(df_pol[subject == "Trump"  & party_code == "R"]$score_nominate, df_pol[subject == "Trump"& party_code == "R"]$score_coded)

nom_v_score <- ggplot(df_pol, aes(y = score_coded, x = score_nominate, color = party_code)) +
                    geom_point() +
                    geom_abline(intercept = 0, slope = 1, color = "black", linewidth = 1, linetype = 23, alpha = 0.5) + 
                    facet_wrap(~subject, nrow = 2) +
                    scale_color_manual(breaks = c("R", "D"),
                                      labels = c("Republican", "Democrat"),
                                      values = c("#ff8080", "#8080ff"),
                                      name = "Party", guide = 'legend') + 
                    geom_smooth(method = 'lm', alpha = 0.3) +
                    lims(y = c(-1, 1),
                         x = c(-1, 1)) +
                    labs(y = "Human-Coded score", x = "DW-Nominate") +
                    theme_bw() +
                    theme(strip.background = element_blank(),
                          legend.position = 'bottom')

ggsave("./figs/Fig S. Coded scores v. Nominate scores.pdf", nom_v_score,
       device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig S. Coded scores v. Nominate scores.eps", height = 8.27, width = 11.69)
nom_v_score
dev.off()

# Scatters indicate that within each part, nominate scores are providing
# very little information about respective tweet codes (slopes are basically 0)

# Generate contingency tables using coded scores:
df_pol[, score_coded_bin := ifelse(score_coded > 0, 1, -1)]

table(df_pol[subject == "Trump"]$score_coded_bin, df_pol[subject == "Trump"]$score)
table(df_pol[subject == "Biden"]$score_coded_bin, df_pol[subject == "Biden"]$score)

######################################
# Predicted v. Observed Scatterplots #
######################################

# Only plot scatters for continuous outcomes using coded scores

plot_pred_v_obs <- function(d_name, vers, dd){
  
  dd <- dd[data_name == d_name,]
  
  if (vers == 'Tuned'){
    dd <- dd[!is.na(tune_name_long)]
    
    dd[!is.na(prompt_name_long) & is.na(tune_name_long), model_name_long := paste0(model_name_long, ": ", prompt_name_long)]
    dd[!is.na(tune_name_long), model_name_long := paste0(model_name_long, ":\n ", tune_name_long)]
    
  }else if (vers == 'Pretrained'){
    dd <- dd[is.na(tune_name_long) & prompt_name_long %in% c("Stance", NA) & model_name_long != "SenticNet 4"]
  }else{
    dd <- dd[!is.na(prompt_name_long) & model_name_long != "GPT-4",]
    dd[, model_name_long := paste0(model_name_long, ": ", prompt_name_long)]
  }
  
  if (d_name == 'pol'){
    dd[, score := score_coded]
    dd <- dd[!is.na(score), ]
  }
  
  subplot <- ggplot(dd, aes(y = score, x = est_score, color = subject)) + 
              geom_abline(intercept = 0, slope = 1, color = "black", linewidth = 1, linetype = 23, alpha = 0.5) + 
              geom_point(alpha = 0.2) + 
              geom_smooth(method = 'lm', linewidth = 1, alpha = 0.2) + 
              scale_color_manual(values = c("#ff8080", "#8080ff"),
                                 breaks = c("Trump", "Biden")) +
              facet_wrap(~model_name_long, ncol = 3) + 
              labs(x = "Estimated score", 
                   y = "Human-Coded \nscores",
                   color = "Subject") +
              theme_bw() + 
              theme(strip.background = element_blank(),
                    legend.position = "bottom",
                    axis.title.y = element_text(angle = 0, vjust = 0.5))
  
  return(subplot)
  
}

args <- expand.grid("vers" = c("Pretrained", "Tuned", "Prompts"),
                    "d_name" = c('pol', 'user'))

p <- mapply(plot_pred_v_obs, args$d_name, args$vers, MoreArgs = list(dd = df), SIMPLIFY = F)

for (i in 1:dim(args)[1]){
  
  filename_pdf <- sprintf("./figs/SA2 Fig %s - %s %s.pdf", i, args[i,]$d_name, args[i, ]$vers)
  filename_eps <- sprintf("./figs/SA2 Fig %s - %s %s.eps", i, args[i,]$d_name, args[i, ]$vers)

  if (args[i, ]$vers == 'Tuned'){
    ggsave(filename_pdf,
           plot = p[[i]], 
           width = 8.27, 
           height = 11.69)
    
    cairo_ps(filename_eps, height = 8.27, width = 11.69)
    p[[i]]
    dev.off()
    
  }else{
    ggsave(filename_pdf,
           plot = p[[i]], 
           width = 8.27, 
           height = 11.69/2)
    
    cairo_ps(filename_eps, height = 8.27, width = 11.69/2)
    p[[i]]
    dev.off()
  
  }
}

############################################
# Prep summary statistics by variable type #
############################################

# Process summary statistics files and create paper plots:
summary_filepath <- "./data/results/summary_statistics/"
files <- paste0(summary_filepath, list.files(summary_filepath))

# For now, leave out the five category test and the nominate test since 
# these results do not show different patterns than existing files:
files <- files[!(files %like% "polnom"|files %like% "user5cat")]

# Create lookup table to add back model_id to make merges easier later:
df_id <- unique(df[, .(model_id, model_name, tuned, tune_data, data_name, prompt_name, subject)])

process_excel <- function(fp){
  
  fp_sheets <- excel_sheets(fp)
  
  d_name <- gsub(".*/||Results.*", "", fp)
  
  files <- list()
  
  for (s in fp_sheets){
    
    d <- as.data.table(read_excel(fp, sheet = s))
    
    # First column name is blank so set to "model_name"
    setnames(d, "...1", "model_name")
    
    # Recover subject column and leave metrics as separate columns
    d <- melt(d, id.vars = c("model_name"), variable.name = 'metric', value.name = 'value')
    d[, subject := gsub("(biden|trump).*", "\\1", metric)]
    d[, metric := gsub(".*\\.", "", metric)]
    
    d <- dcast(d, model_name + subject ~ metric, value.var = "value")
    
    # Recover data on prompt name and dataset used for tuning:
    extract_tune <- ".*\\.(handcode|nominate|party).*"
    d[, tune_data := ifelse(grepl(extract_tune, model_name), 
                            gsub(extract_tune, "\\1", model_name), NA)]
    
    extract_prompt <- ".*\\.(p[0-9]{1})\\.*"
    d[, prompt_name := ifelse(grepl(extract_prompt, model_name), 
                            gsub(extract_prompt, "\\1", model_name), NA)]
    
    d[, model_name := gsub("\\..*", "", model_name)]
    d[, data_name := d_name]
    
    d[, outcome_type := tolower(s)]
    d[, tuned := ifelse(is.na(tune_data), F, T)]
    
    # For user data, we have summary statistics for the entire dataset,
    # for rows with single subjects, and rows with boths subjects. Add additional
    # indicator column for this.
    
    if (d_name == 'user'){
      d[, sub_pop := gsub(".*, ", "", outcome_type)]
      d[, outcome_type := gsub(", .*", "", outcome_type)]
    }else{
      d[, sub_pop := NA]
    }
    
    # Merge on model_id from primary dataframe:
    d[, subject := str_to_title(subject)]
    d <- join(d, df_id, type = 'left', by = c("model_name", "tuned", "tune_data", "data_name", "prompt_name", "subject"))
    
    # Merge on processed names:
    long_names <- unique(df[, .(model_id, model_name_long, data_name_long, prompt_name_long, tune_name_long)])
    d <- join(d, long_names, type = 'left', by = "model_id")
    
    # Reorder column names
    first_cols <- c("model_name", "model_name_long", "tuned", "tune_data",  "tune_name_long",
                    "data_name", "data_name_long", "prompt_name", "prompt_name_long", "subject", "outcome_type", "sub_pop")
    remaining_cols <- names(d)[!(names(d) %in% first_cols)]
    d <- d[, c(first_cols, remaining_cols), with = F]
    
    files[[s]] <- d
  }
  
  return(files)
}

df_summary <- lapply(files, process_excel)

# Separate out files by outcome type:
df_bin    <- do.call(rbind, lapply(df_summary, function(x) rbindlist(x[grep("Binary", names(x))])))
df_cont   <- do.call(rbind, lapply(df_summary, function(x) rbindlist(x[grep("Continuous", names(x))])))
df_cat    <- do.call(rbind, lapply(df_summary, function(x) rbindlist(x[grep("Categorical", names(x))])))

# Set up a select series of models we'll use throughout multiple figures:

#########################################################
# Fig 2. Mean scores by party and correlation, pol data #
#########################################################

keep_models <- c("gpt4o", "gpt35", "deberta", "siebert")

# For Fig 2, calculate mean estimated score for each model in the politican dataset:
df_fig2 <- df[data_name == 'pol', .(est_score, model_id, id)]
df_fig2 <- join(df_fig2, pol_id[, .(id, party_code, subject)], type = 'left')
df_fig2 <- df_fig2[, mean(.SD$est_score), by = c("model_id", "party_code", "subject")]
setnames(df_fig2, "V1", "est_mean")
df_fig2[, subject := str_to_title(subject)]

df_fig2 <- join(df_cont, df_fig2, by = c("model_id", "subject"), type = "right")
df_fig2 <- dcast(df_fig2, model_name + model_name_long + tuned + tune_data + data_name + prompt_name + subject + r + lower + upper ~ party_code, value.var = c("est_mean"))
df_fig2 <- df_fig2[tuned == F & prompt_name %in% c(NA, "p3") & model_name %in% keep_models,]

# Get a unique vector of model names, then sort that vector by the highest correlation
# within the Biden models
levs <- unique(df_fig2[subject == "Biden",]$model_name_long[order(df_fig2[subject == "Biden",]$r, decreasing = F)])
df_fig2[, model_name_long := factor(model_name_long, levels = levs)]

plot_colors <- c("mean_repub" = "#ff8080", "mean_dem" = "#8080ff")

fig_2a <- ggplot(df_fig2, aes(y = model_name_long)) +
                geom_linerange(aes(xmin = R, xmax = D,
                                   y = model_name_long),
                               color = "black", alpha = 0.3, linewidth = 1) +
                geom_point(aes(x = R, color = "mean_repub"), size = 4) +
                geom_point(aes(x = D, color = "mean_dem"), size = 4) +
                facet_wrap(~subject, nrow = 2) +
                labs(x = "\nEstimated Mean Stance", 
                     y = "",
                     color = "Party identification") +
                theme_bw() +
                scale_color_manual(values = plot_colors, 
                                   labels = c("Democrat", "Republican")) +
                scale_x_continuous(breaks = seq(-1, 1, 0.5),
                                   limits = c(-1, 1)) +
                theme(axis.line.x = element_line(colour = "black", size = 0.65),
                      axis.ticks.x = element_line(size = 0.65),
                      axis.ticks.length = unit(2, "mm"),
                      axis.text = element_text(size = 12, family = "sans"),
                      axis.title.y = element_text(angle = 0, vjust = 0.5),
                      axis.title = element_text(size = 14, family = "sans"),
                      strip.background = element_blank(),
                      strip.text = element_blank(),
                      plot.title = element_blank(),
                      panel.spacing.y = unit(0.5, "cm"),
                      legend.position = "bottom",
                      legend.text = element_text(size = 10, family = 'sans'))

fig_2b <- ggplot(df_fig2, aes(y = model_name_long)) +
                  geom_point(aes(x = r), color = "black", size = 2) +
                  geom_linerange(aes(xmin = lower, xmax = upper),
                                 color = "black", alpha = 0.6, linewidth = 1) +
                  labs(y = "",
                       x = "\nCorrelation") +
                  theme_bw() +
                  scale_x_continuous(breaks = seq(0, 1, 0.2),
                                     limits = c(0, 1)) +
                  facet_wrap(~subject, nrow = 2, strip.position = "right") +
                  theme(axis.line.x = element_line(colour = "black", size = 0.65),
                        axis.ticks.x = element_line(size = 0.65),
                        axis.ticks.length = unit(2, "mm"),
                        axis.ticks.y = element_blank(),
                        axis.title.y = element_blank(),
                        axis.text.y = element_blank(),
                        axis.text = element_text(size = 12, family = "sans"),
                        axis.title = element_text(size = 14, family = "sans"),
                        strip.background = element_blank(),
                        strip.text = element_text(size = 14, family = "sans"),
                        plot.title = element_blank(),
                        panel.spacing.y = unit(0.5, "cm"))

page <- ggarrange(fig_2a, fig_2b, ncol = 2, common.legend = T, legend = "bottom",
                  widths = c(1.2, 1))

ggsave(plot = page, filename = "./figs/Fig 2.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 2.eps", height = 8.27, width = 11.69)
page
dev.off()

##########################################################
# Fig 3. Correlation, user data, single or both subjects #
##########################################################

keep_models <- c("gpt4o", "gpt35", "deberta", "siebert")

df_fig3 <- df_cont[model_name %in% keep_models & tuned == F & prompt_name %in% c(NA, "p3") & data_name == 'user']

# Get a unique vector of model names, then sort that vector by the highest correlation
# within the Biden models
levs <- unique(df_fig3[subject == "Biden" & sub_pop == 'one',]$model_name_long[order(df_fig3[subject == "Biden" & sub_pop == 'one',]$r, decreasing = F)])
df_fig3[, model_name_long := factor(model_name_long, levels = levs)]

# Remove estimates over entire set of user tweets:
df_fig3 <- df_fig3[sub_pop != 'all']
df_fig3[, number_subjects := ifelse(sub_pop == "both", "Both", "Single")]

plot_colors <- c("Single" = "#b85e00", "Both" =  "#1b3644") 

fig_3 <- ggplot(df_fig3, aes(y = model_name_long)) +
          geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
          geom_pointrange(aes(x = r, xmin = lower, xmax = upper, 
                              color = number_subjects), 
                          position = position_dodge2v(height = 0.4, reverse = T)) +
          labs(y = "",
               x = "Correlation",
               color = "Number of subjects mentioned") +
          theme_bw() +
          coord_fixed(ratio = 0.5, xlim = c(-0.5, 1), ylim = c(1, 4)) +
          scale_x_continuous(breaks = seq(-0.5, 1, 0.5)) +
          scale_color_manual(values = plot_colors) +
          facet_wrap(~subject) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                strip.background = element_blank(),
                strip.text = element_text(size = 14, family = "sans", angle = 0),
                plot.margin = unit(c(0,0,0,0), "cm"),
                panel.spacing.y = unit(1, "cm"),
                legend.position = "bottom",
                legend.text = element_text(size = 10, family = 'sans'))

ggsave(plot = fig_3, filename = "./figs/Fig 3.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 3.eps", height = 8.27, width = 11.69)
fig_3
dev.off()

################################################## 
# Fig 4. Changes in correlation following tuning #
##################################################

keep_models <- c("gpt4o", "gpt35", "deberta", 'siebert')

# Compare results in Pol and User datasets:
# Pretrained Deberta, GPT3.5, GPT4o
# v. Tuned w/ party, tuned with handcode, or using few-shot prompt
df_fig4 <- df_cont[data_name %in% c('pol', 'user') & 
                   model_name %in% keep_models & 
                   ((prompt_name %in% c(NA, "p3"))|(tune_data %in% c('party', "handcode"))),]

# For user-data, hold onto results for single or multiple (not whole sample).
# Remove tuning by DW-Nominate to simplify figure
df_fig4 <- df_fig4[sub_pop %in% c(NA, "both", "one"),]
df_fig4 <- df_fig4[tune_data %in% c(NA, "party", "handcode")]

# How was model tuned?
df_fig4[, tuning := ifelse(!is.na(tune_data), as.character(tune_name_long), "Pretrained")]
df_fig4[, tuning := factor(tuning, c("Pretrained", "Tuned (Party Affiliation)", "Tuned (Human-Coded)"))]

# Create plot facets for populations:
df_fig4[, population := ifelse(data_name == 'pol', "Politicians", 
                               ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)"))]

df_fig4[, population := factor(population, levels = c("Politicians", "Users \n(Single subject)", "Users \n(Both subjects)"))]

plot_colors <- c("Pretrained" = "#a6611a", 
                 "Tuned (Party Affiliation)" = "#dfc27d",
                "Tuned (Human-Coded)" = "#018571")

fig_4 <- ggplot(df_fig4, aes(y = model_name_long, x = r, color = tuning)) +
          geom_point(size = 2, position = position_dodge2(width = 0.5)) +
          geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
          labs(y = "",
               x = "Correlation",
               color = "") +
          theme_bw() +
          coord_fixed(ratio = 0.2, xlim = c(-0.2, 1)) +
          scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
          facet_grid(population~subject) +
          scale_color_manual(values = plot_colors) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                axis.title.y = element_text(size = 12, family = "sans"),
                axis.title.x = element_text(size = 12, family = "sans", 
                                            vjust = -0.5),
                strip.background = element_blank(),
                strip.text = element_text(size = 12, family = "sans"),
                legend.position = "bottom",
                legend.justification = "center",
                legend.text = element_text(size = 12, family = 'sans'),
                panel.spacing.x = unit(0.5, "cm"),
                panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_4, filename = "./figs/Fig 4a.pdf", device = 'pdf', bg = "transparent",
       height = 8.27*1.5, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 4a.eps", height = 8.27*1.5, width = 11.69)
fig_4
dev.off()

###################################################################
# Fig 5. Changes in correlation following tuning - Binary version #
###################################################################

df_fig5 <- df_bin[data_name %in% c('pol', 'user') & 
                     model_name %in% keep_models & 
                     ((prompt_name %in% c(NA, "p4"))|(tune_data %in% c('party', "handcode"))),]

# For user-data, hold onto results for single or multiple (not whole sample).
# Remove tuning by DW-Nominate to simplify figure
df_fig5 <- df_fig5[sub_pop %in% c(NA, "both", "one"),]
df_fig5 <- df_fig5[tune_data %in% c(NA, "party", "handcode")]

# How was model tuned?
df_fig5[, tuning := ifelse(!is.na(tune_data), as.character(tune_name_long), "Pretrained")]
df_fig5[, tuning := factor(tuning, c("Pretrained", "Tuned (Party Affiliation)", "Tuned (Human-Coded)"))]

# Create plot facets for populations:
df_fig5[, population := ifelse(data_name == 'pol', "Politicians", 
                               ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)"))]

df_fig5[, population := factor(population, levels = c("Politicians", "Users \n(Single subject)", "Users \n(Both subjects)"))]

plot_colors <- c("Pretrained" = "#a6611a", 
                 "Tuned (Party Affiliation)" = "#dfc27d",
                 "Tuned (Human-Coded)" = "#018571")

fig_5 <- ggplot(df_fig5, aes(y = model_name_long, x = r, color = tuning)) +
    geom_point(size = 2, position = position_dodge2(width = 0.5)) +
    geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
    labs(y = "",
         x = "Correlation",
         color = "") +
    theme_bw() +
    coord_fixed(ratio = 0.2, xlim = c(-0.2, 1), ylim = c(1, 4)) +
    scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
    facet_grid(population~subject) +
    scale_color_manual(values = plot_colors) +
    theme(axis.line.x = element_line(colour = "black", size = 0.65),
          axis.ticks.x = element_line(size = 0.65),
          axis.ticks.length = unit(2, "mm"),
          axis.text = element_text(size = 12, family = "sans"),
          plot.title = element_blank(),
          axis.title.y = element_text(size = 12, family = "sans"),
          axis.title.x = element_text(size = 12, family = "sans", 
                                      vjust = -0.5),
          strip.background = element_blank(),
          strip.text = element_text(size = 12, family = "sans"),
          legend.position = "bottom",
          legend.justification = "center",
          legend.text = element_text(size = 12, family = 'sans'),
          panel.spacing.x = unit(0.5, "cm"),
          panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_5, filename = "./figs/Fig 5a.pdf", device = 'pdf', bg = "transparent",
       height = 8.27*1.5, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 5a.eps", height = 8.27*1.5, width = 11.69)
fig_5
dev.off()

################################################
# Fig 6. How do models perform across prompts? #
################################################

keep_models <- c('gpt35', 'gpt4o')

df_fig6 <- df_cont[data_name %in% c('pol', 'user') & 
                  model_name %in% keep_models & 
                  (tuned == F|tuned == T & tune_data %in% c("handcode", "party")),]

# Hold onto tuned results for in-target tuning
df_fig6 <- df_fig6[((tuned == F)|(tune_data == "party" & data_name == "pol")|(tune_data == "handcode" & data_name == "user")),]

df_fig6 <- df_fig6[sub_pop %in% c(NA, "both", "one"),]

df_fig6[, population := ifelse(data_name == 'pol', "Politicians", 
                               ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)"))]

df_fig6[, population := factor(population, levels = c("Politicians", "Users \n(Single subject)", "Users \n(Both subjects)"))]

# For tune data, change prompt name:
df_fig6[tuned == T, prompt_name_long := "Stance (Tuned)"]
df_fig6[tuned == T, prompt_name := "p7"]

df_fig6[, prompt_name := str_to_title(prompt_name)]
df_fig6[, prompt_name := factor(prompt_name, levels = sort(unique(df_fig6$prompt_name)))]

# Remove the alternative prompt and binary prompt from this figure:
df_fig6 <- df_fig6[!(prompt_name_long %in% c("Stance (Binary)", "Stance (Alt)")),]

# plot_colors <- c("p1" = "#ccebc5",
#                  "p2" = "#a8ddb5",
#                  'p3' = "#7bccc4",
#                  'p4' = "#4eb3d3",
#                  "p5" = "#2b8cbe",
#                  'p6' = "#08589e")

plot_colors <- c("GPT-3.5 Turbo" = '#8dd3c7',
                 "GPT-4" = '#bebada', 
                 "GPT-4 Omni" = '#fb8072')

fig_6 <- ggplot(df_fig6, aes(x = r, y = prompt_name_long, color = model_name_long)) +
            geom_vline(xintercept = 0, linewidth = 0.5, linetype = 21, alpha = 0.5) +
            geom_point(size = 2, position = position_dodge2(width = 0.6)) +
            geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
            labs(x = "Correlation",
                 y = "Prompt",
                 color = "Method") +
            theme_bw() +
            scale_x_continuous(breaks = seq(-0.4, 1, 0.2),
                               limits = c(-0.4, 1)) +
            facet_grid(population~subject) +
            scale_color_manual(values = plot_colors) +
            theme(axis.line.x = element_line(colour = "black", size = 0.65),
                  axis.ticks.x = element_line(size = 0.65),
                  axis.ticks.length = unit(2, "mm"),
                  axis.text = element_text(size = 12, family = "sans"),
                  plot.title = element_blank(),
                  axis.title.y = element_text(size = 12, family = "sans"),
                  axis.title.x = element_text(size = 12, family = "sans", 
                                              vjust = -0.5),
                  strip.background = element_blank(),
                  strip.text = element_text(size = 12, family = "sans"),
                  legend.position = "bottom",
                  legend.justification = "center",
                  legend.title = element_text(size = 12, family = 'sans'),
                  legend.text = element_text(size = 10, family = 'sans'),
                  panel.spacing.x = unit(0.5, "cm"),
                  panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_6, filename = "./figs/Fig 6.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 6.eps", height = 8.27, width = 11.69)
fig_6
dev.off()

#################################################################
# Fig 7. How do models perform across prompts? (Binary version) #
#################################################################

keep_models <- c('gpt35', 'gpt4o')

df_fig7 <- df_bin[data_name %in% c('pol', 'user') & 
                     model_name %in% keep_models & 
                     tuned == F,]

df_fig7 <- df_fig7[sub_pop %in% c(NA, "both", "one"),]

df_fig7[, population := ifelse(data_name == 'pol', "Politicians", 
                               ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)"))]

df_fig7[, population := factor(population, levels = c("Politicians", "Users \n(Single subject)", "Users \n(Both subjects)"))]

df_fig7[, prompt_name := str_to_title(prompt_name)]
df_fig7[, prompt_name := factor(prompt_name, levels = sort(unique(df_fig7$prompt_name)))]

df_fig7 <- df_fig7[!(prompt_name_long %in% c("Stance", "Stance (Alt)")),]

# plot_colors <- c("p1" = "#ccebc5",
#                  "p2" = "#a8ddb5",
#                  'p3' = "#7bccc4",
#                  'p4' = "#4eb3d3",
#                  "p5" = "#2b8cbe",
#                  'p6' = "#08589e")

plot_colors <- c("GPT-3.5 Turbo" = '#8dd3c7',
                 "GPT-4" = '#bebada', 
                 "GPT-4 Omni" = '#fb8072')

fig_7 <- ggplot(df_fig7, aes(x = r, y = prompt_name_long, color = model_name_long)) +
          geom_vline(xintercept = 0, linewidth = 0.5, linetype = 21, alpha = 0.5) +
          geom_point(size = 2, position = position_dodge2(width = 0.6)) +
          geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
          labs(x = "Correlation",
               y = "Prompt",
               color = "Method") +
          theme_bw() +
          scale_x_continuous(breaks = seq(-0.4, 1, 0.2),
                             limits = c(-0.4, 1)) +
          facet_grid(population~subject) +
          scale_color_manual(values = plot_colors) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                axis.title.y = element_text(size = 12, family = "sans"),
                axis.title.x = element_text(size = 12, family = "sans", 
                                            vjust = -0.5),
                strip.background = element_blank(),
                strip.text = element_text(size = 12, family = "sans"),
                legend.position = "bottom",
                legend.justification = "center",
                legend.title = element_text(size = 12, family = 'sans'),
                legend.text = element_text(size = 10, family = 'sans'),
                panel.spacing.x = unit(0.5, "cm"),
                panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_7, filename = "./figs/Fig 7.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 7.eps", height = 8.27, width = 11.69)
fig_7
dev.off()


#############################################
# Appendix - Fig 8/9: Categorical Fig 3 & 4 #
#############################################

####### Alt Fig 3 w/ categorical outcome

keep_models <- c("gpt4o", "gpt35", "siebert", "vader", "nrc")

df_fig8 <- df_cat[model_name %in% keep_models & tuned == F & prompt_name %in% c(NA, "p3") & data_name == 'user']

# Get a unique vector of model names, then sort that vector by the highest correlation
# within the Biden models
levs <- unique(df_fig8[subject == "Biden",]$model_name_long[order(df_fig8[subject == "Biden",]$CramerV, decreasing = F)])
df_fig8[, model_name_long := factor(model_name_long, levels = levs)]

# Remove estimates over entire set of user tweets:
df_fig8 <- df_fig8[sub_pop != 'all']
df_fig8[, number_subjects := ifelse(sub_pop == "both", "Both", "Single")]

plot_colors <- c("Single" = "#b85e00", "Both" =  "#1b3644") 

fig_8 <- ggplot(df_fig8, aes(y = model_name_long)) +
          geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
          geom_point(aes(x = CramerV, color = number_subjects), 
                          position = position_dodge2v(height = 0.4, reverse = T)) +
          labs(y = "",
               x = "Correlation",
               color = "Number of subjects mentioned") +
          theme_bw() +
          coord_fixed(ratio = 0.5, xlim = c(-0.5, 1), ylim = c(1, 5)) +
          scale_x_continuous(breaks = seq(-0.5, 1, 0.5)) +
          scale_color_manual(values = plot_colors) +
          facet_wrap(~subject) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                strip.background = element_blank(),
                strip.text = element_text(size = 14, family = "sans", angle = 0),
                plot.margin = unit(c(0,0,0,0), "cm"),
                panel.spacing.y = unit(1, "cm"),
                legend.position = "bottom",
                legend.text = element_text(size = 10, family = 'sans'))

ggsave(plot = fig_8, filename = "./figs/SA3 Fig 1 - Fig 3 using cat.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/SA3 Fig 1 - Fig 3 using cat.eps", height = 8.27, width = 11.69)
fig_8
dev.off()

####### Alt Fig 4 w/ categorical outcome

keep_models <- c("gpt4o", "gpt35", 'siebert')

# Compare results in Pol and User datasets:
# Pretrained Deberta, GPT3.5, GPT4o
# v. Tuned w/ party, tuned with handcode, or using few-shot prompt
df_fig9 <- df_cat[data_name %in% c('user') & 
                     model_name %in% keep_models & 
                     ((prompt_name %in% c(NA, "p3"))|(tune_data %in% c("handcode"))),]

# For user-data, hold onto results for single or multiple (not whole sample).
# Remove tuning by DW-Nominate to simplify figure
df_fig9 <- df_fig9[sub_pop %in% c(NA, "both", "one"),]
df_fig9 <- df_fig9[tune_data %in% c(NA, "handcode")]

# How was model tuned?
df_fig9[, tuning := ifelse(is.na(tune_data), "Pretrained", "Tuned (Human-Coded)")]
df_fig9[, tuning := factor(tuning, c("Pretrained", "Tuned (Human-Coded)"))]

# Create plot facets for populations:
df_fig9[, population := ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)")]
df_fig9[, population := factor(population, levels = c("Users \n(Single subject)", "Users \n(Both subjects)"))]

plot_colors <- c("Pretrained" = "#a6611a", 
                 "Tuned (Human-Coded)" = "#018571")

fig_9 <- ggplot(df_fig9, aes(y = model_name_long, x = CramerV, color = tuning)) +
          geom_point(size = 2, position = position_dodge2(width = 0.5)) +
          labs(y = "",
               x = "Correlation",
               color = "") +
          theme_bw() +
          coord_fixed(ratio = 0.2, xlim = c(-0.2, 1), ylim = c(1, 3)) +
          scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
          facet_grid(population~subject) +
          scale_color_manual(values = plot_colors) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                axis.title.y = element_text(size = 12, family = "sans"),
                axis.title.x = element_text(size = 12, family = "sans", 
                                            vjust = -0.5),
                strip.background = element_blank(),
                strip.text = element_text(size = 12, family = "sans"),
                legend.position = "bottom",
                legend.justification = "center",
                legend.text = element_text(size = 12, family = 'sans'),
                panel.spacing.x = unit(0.5, "cm"),
                panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_9, filename = "./figs/SA3 Fig 2 - Fig 4 using cat.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/SA3 Fig 2 - Fig 4 using cat.eps", height = 8.27, width = 11.69)
fig_9
dev.off()

#############################################################
# Alt Fig 4 - Using Nominate scores instead of party scores #
#############################################################

keep_models <- c("gpt4o", "gpt35", "deberta", 'siebert')

# Compare results in Pol and User datasets:
# Pretrained Deberta, GPT3.5, GPT4o
# v. Tuned w/ party, tuned with handcode, or using few-shot prompt
df_fig10 <- df_cont[data_name %in% c('pol', 'user') & 
                     model_name %in% keep_models & 
                     ((tune_data %in% c('nominate', "handcode", "party"))),]

# For user-data, hold onto results for single or multiple (not whole sample).
# Remove tuning by DW-Nominate to simplify figure
df_fig10 <- df_fig10[sub_pop %in% c(NA, "both", "one"),]
df_fig10 <- df_fig10[tune_data %in% c("party", "nominate", "handcode")]

# How was model tuned?
df_fig10[, tuning := as.character(tune_name_long)]
df_fig10[, tuning := factor(tuning, c("Tuned (Party Affiliation)", "Tuned (DW-Nominate)", "Tuned (Human-Coded)"))]

# Create plot facets for populations:
df_fig10[, population := ifelse(data_name == 'pol', "Politicians", 
                               ifelse(sub_pop == 'both', "Users \n(Both subjects)", "Users \n(Single subject)"))]

df_fig10[, population := factor(population, levels = c("Politicians", "Users \n(Single subject)", "Users \n(Both subjects)"))]

plot_colors <- c("Tuned (Party Affiliation)" = "#a6611a", 
                 "Tuned (DW-Nominate)" = "#dfc27d",
                 "Tuned (Human-Coded)" = "#018571")

fig_10 <- ggplot(df_fig10, aes(y = model_name_long, x = r, color = tuning)) +
          geom_point(size = 2, position = position_dodge2(width = 0.6)) +
          geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.6)) +
          labs(y = "",
               x = "Correlation",
               color = "") +
          theme_bw() +
          coord_fixed(ratio = 0.2, xlim = c(-0.2, 1), ylim = c(1, 4)) +
          scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
          facet_grid(population~subject) +
          scale_color_manual(values = plot_colors) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                axis.title.y = element_text(size = 12, family = "sans"),
                axis.title.x = element_text(size = 12, family = "sans", 
                                            vjust = -0.5),
                strip.background = element_blank(),
                strip.text = element_text(size = 12, family = "sans"),
                legend.position = "bottom",
                legend.justification = "center",
                legend.text = element_text(size = 12, family = 'sans'),
                panel.spacing.x = unit(0.5, "cm"),
                panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_10, filename = "./figs/SA3 Fig 3 - Fig 4 using dw-nominate.pdf", device = 'pdf', bg = "transparent",
       height = 8.27*2, width = 11.69, unit = 'in')

cairo_ps("./figs/SA3 Fig 3 - Fig 4 using dw-nominate.eps", height = 8.27, width = 11.69)
fig_10
dev.off()

#######################################
# Fig 3 - 5: Using Li & Kawintiranon #
#######################################

######### Fig 3 - Li and Kaw

keep_models <- c("gpt4o", "gpt35", "deberta", "siebert", "vader", "nrc")

df_fig11 <- df_cont[model_name %in% keep_models & tuned == F & prompt_name %in% c(NA, "p3") & data_name %in% c("li", "kawintiranon")]

# Get a unique vector of model names, then sort that vector by the highest correlation
# within the Biden models & Li
levs <- unique(df_fig11[subject == "Biden" & data_name == "li",]$model_name_long[order(df_fig11[subject == "Biden" & data_name == "li",]$r, decreasing = F)])
df_fig11[, model_name_long := factor(model_name_long, levels = levs)]

df_fig11[, population := str_to_title((as.character(data_name)))]
df_fig11[, population := factor(population, levels = c("Li", "Kawintiranon"))]

fig_11 <- ggplot(df_fig11, aes(y = model_name_long)) +
          geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
          geom_pointrange(aes(x = r, xmin = lower, xmax = upper), 
                          position = position_dodge2v(height = 0.4, reverse = T)) +
          labs(y = "",
               x = "Correlation") +
          theme_bw() +
          coord_fixed(ratio = 0.25, xlim = c(-0.5, 1), ylim = c(1, 6)) +
          scale_x_continuous(breaks = seq(-0.5, 1, 0.5)) +
          facet_grid(population~subject) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                strip.background = element_blank(),
                strip.text = element_text(size = 14, family = "sans", angle = 0),
                plot.margin = unit(c(0,0,0,0), "cm"),
                panel.spacing.y = unit(1, "cm"),
                legend.position = "bottom",
                legend.text = element_text(size = 10, family = 'sans'))

ggsave(plot = fig_11, filename = "./figs/Fig 3 - Li & Kaw.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 3 - Li & Kaw.eps", height = 8.27, width = 11.69)
fig_11
dev.off()


######### Fig 4 - Li and Kaw

keep_models <- c("gpt4o", "gpt35", "deberta", 'siebert')

# Compare results in Pol and User datasets:
# Pretrained Deberta, GPT3.5, GPT4o
# v. Tuned w/ party, tuned with handcode, or using few-shot prompt
df_fig12 <- df_cont[data_name %in% c("li", "kawintiranon") & 
                     model_name %in% keep_models & 
                     ((prompt_name %in% c(NA, "p3"))|(tune_data %in% c('party', "handcode"))),]

# Remove tuning by DW-Nominate to simplify figure
df_fig12 <- df_fig12[tune_data %in% c(NA, "party", "handcode")]

# How was model tuned?
df_fig12[, tuning := ifelse(!is.na(tune_data), as.character(tune_name_long), "Pretrained")]
df_fig12[, tuning := factor(tuning, c("Pretrained", "Tuned (Party Affiliation)", "Tuned (Human-Coded)"))]

# Create plot facets for populations:
df_fig12[, population := ifelse(data_name == 'li', "Li", "Kawintiranon")]

df_fig12[, population := factor(population, levels = c("Li", "Kawintiranon"))]

plot_colors <- c("Pretrained" = "#a6611a", 
                 "Tuned (Party Affiliation)" = "#dfc27d",
                 "Tuned (Human-Coded)" = "#018571")

fig_12 <- ggplot(df_fig12, aes(y = model_name_long, x = r, color = tuning)) +
          geom_point(size = 2, position = position_dodge2(width = 0.5)) +
          geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
          labs(y = "",
               x = "Correlation",
               color = "") +
          theme_bw() +
          coord_fixed(ratio = 0.2, xlim = c(-0.2, 1), ylim = c(1, 4)) +
          scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
          facet_grid(population~subject) +
          scale_color_manual(values = plot_colors) +
          theme(axis.line.x = element_line(colour = "black", size = 0.65),
                axis.ticks.x = element_line(size = 0.65),
                axis.ticks.length = unit(2, "mm"),
                axis.text = element_text(size = 12, family = "sans"),
                plot.title = element_blank(),
                axis.title.y = element_text(size = 12, family = "sans"),
                axis.title.x = element_text(size = 12, family = "sans", 
                                            vjust = -0.5),
                strip.background = element_blank(),
                strip.text = element_text(size = 12, family = "sans"),
                legend.position = "bottom",
                legend.justification = "center",
                legend.text = element_text(size = 12, family = 'sans'),
                panel.spacing.x = unit(0.5, "cm"),
                panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_12, filename = "./figs/Fig 4 - Li and Kaw.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 4 - Li and Kaw.eps", height = 8.27, width = 11.69)
fig_12
dev.off()

#### Fig 6 - Kaw and Li:

keep_models <- c('gpt35', 'gpt4o')

df_fig13 <- df_cont[data_name %in% c("li", "kawintiranon") & 
                     model_name %in% keep_models & 
                     tuned == F,]

df_fig13[, population := ifelse(data_name == 'li', "Li", "Kawintiranon")]

df_fig13[, population := factor(population, levels = c("Li", "Kawintiranon"))]

df_fig13[, prompt_name := str_to_title(prompt_name)]
df_fig13[, prompt_name := factor(prompt_name, levels = sort(unique(df_fig13$prompt_name)))]

df_fig13 <- df_fig13[!prompt_name %in% c("P4", "P2")]

# plot_colors <- c("p1" = "#ccebc5",
#                  "p2" = "#a8ddb5",
#                  'p3' = "#7bccc4",
#                  'p4' = "#4eb3d3",
#                  "p5" = "#2b8cbe",
#                  'p6' = "#08589e")

plot_colors <- c("GPT-3.5 Turbo" = '#8dd3c7',
                 "GPT-4" = '#bebada', 
                 "GPT-4 Omni" = '#fb8072')

fig_13 <- ggplot(df_fig13, aes(x = r, y = prompt_name_long, color = model_name_long)) +
            geom_vline(xintercept = 0, linewidth = 0.5, linetype = 21, alpha = 0.5) +
            geom_point(size = 2, position = position_dodge2(width = 0.6)) +
            geom_linerange(aes(xmin = lower, xmax = upper), linewidth = 0.5, alpha = 0.8,  position = position_dodge2(width = 0.5)) +
            labs(y = "Correlation",
                 x = "Prompt",
                 color = "Method") +
            theme_bw() +
            scale_x_continuous(breaks = seq(-0.4, 1, 0.2),
                               limits = c(-0.4, 1)) +
            facet_grid(population~subject) +
            scale_color_manual(values = plot_colors) +
            theme(axis.line.x = element_line(colour = "black", size = 0.65),
                  axis.ticks.x = element_line(size = 0.65),
                  axis.ticks.length = unit(2, "mm"),
                  axis.text = element_text(size = 12, family = "sans"),
                  plot.title = element_blank(),
                  axis.title.y = element_text(size = 12, family = "sans"),
                  axis.title.x = element_text(size = 12, family = "sans", 
                                              vjust = -0.5),
                  strip.background = element_blank(),
                  strip.text = element_text(size = 12, family = "sans"),
                  legend.position = "bottom",
                  legend.justification = "center",
                  legend.title = element_text(size = 12, family = 'sans'),
                  legend.text = element_text(size = 10, family = 'sans'),
                  panel.spacing.x = unit(0.5, "cm"),
                  panel.spacing.y = unit(0.5, "cm"))

ggsave(plot = fig_13, filename = "./figs/Fig 6 - Kaw and Li.pdf", device = 'pdf', bg = "transparent",
       height = 8.27, width = 11.69, unit = 'in')

cairo_ps("./figs/Fig 6 - Kaw and Li.eps", height = 8.27, width = 11.69)
fig_13
dev.off()


#################################
# Supplementary appendix tables #
#################################

# For each measure, separate out tables into pretrained models, tuned models
# and prompt versions, similar to figure results. Tables are hierarchical
# by stance target. Further break these results out by dataset.

# Using one of the result datasets to obtain model ids for each set of
# results:

pretrained_mods <- df_cont[tuned == F & (is.na(prompt_name)|prompt_name == 'p3'), unique(model_id)]
tuned_mods      <- df_cont[tuned == T, unique(model_id)]
prompt_mods     <- df_cont[tuned == F & !is.na(prompt_name), unique(model_id)]

table_sets <- list("Pretrained" = pretrained_mods,
                   "Tuned" = tuned_mods,
                   "Prompts" = prompt_mods)

df_names <- c("116th U.S. Congress", "Twitter Users", "P-Stance", "Kawintiranon & Singh, 2021")
results <- list("Continuous" = df_cont,
                "Binary" = df_bin,
                "Categorical" = df_cat)

filename <- sprintf("./paper/latex/supplementary_tables.tex")

# Overwrite existing file:
con <- file(filename, open = 'w')
close(con)

con <- file(filename, open = 'a')

gen_tables <- function(variable_type, d_name, set){
  
  df_table <- results[[variable_type]]
  
  df_table <- df_table[model_id %in% table_sets[[set]] & data_name_long == d_name, ]
  
  # Write subsection title if this is the first table
  # for a given variable type:
  if ((d_name == "116th U.S. Congress" & set == "Pretrained")|(set == "Categorical" & d_name == "Twitter Users" & set == "Pretrained")){
    writeLines(sprintf("\\subsection{Results using %s benchmark scores}\n", tolower(variable_type)), con)
  }

  # Make sure a dataset name appears correctly in Latex:
  if (d_name == "Kawintiranon & Singh, 2021"){
    d_name <- "Kawintiranon \\& Singh, 2021"
  }
  
  # Add subsubsection for data name:
  if (set == "Pretrained"){
    writeLines(sprintf("\\subsubsection{%s}", d_name), con)
  }
  
  # Set metrics used for each measure:
  if (variable_type == "Continuous"){
    metrics         <- c("MAE", "rMSE", "r", "lower", "upper")
    metrics_renamed <- c("MAE", "rMSE", '$\\rho$')
  }else if (variable_type == "Binary"){
    metrics <- c("F1", "Accuracy", "Precision", "Recall", "r", "lower", "upper")
    metrics_renamed <- c("$F_1$", "Accuracy", "Precision", "Recall", '$\\rho$')
  }else{
    metrics <- c("ChiSq", "CramerV", "Positive", "Neutral", "Negative")
    metrics_renamed <- c("$\\chi^2$", "$\\phi_c$", "$\\rho_{Positive}$", "$\\rho_{Neutral}$", "$\\rho_{Negative}$")
  }
  
  if (set == "Pretrained"){
    keep <- c("model_name_long", "subject")
    keep_rename <- c("")
  }else if (set == "Tuned"){
    keep <- c("model_name_long", "tune_name_long", "subject")
    keep_rename <- c("", "Training Data")
    
    df_table[, tune_name_long := gsub("Tuned |\\(|\\)", "", tune_name_long)]
    df_table[, tune_name_long := factor(tune_name_long, levels = c("Human-Coded", "Party Affiliation", "DW-Nominate"))]
  }else{
    keep <- c("model_name_long", "prompt_name_long", "subject")
    keep_rename <- c("", "Prompt")
  }
  
  if (d_name == "Twitter Users"){
    keep <- c(keep, "sub_pop")
    keep_rename <- c(keep_rename, "\\# Targets")
    
    df_table <- df_table[sub_pop != "all"]
    df_table[, sub_pop := ifelse(sub_pop == "one", "Single", "Multiple")]
    df_table[, sub_pop := factor(sub_pop, levels = c("Single", "Multiple"))]
  }
  
  df_table <- df_table[, c(keep, metrics), with = F]
  
  # Round metrics to 3 decimal places to improve readability
  for (m in metrics){
    df_table[, (m) := round(get(m), 3)]
  }
  
  # Combine CI with mean values:
  if ('r' %in% metrics){
    df_table[, Correlation := paste0(r, " (", lower, ", ", upper, ")")]
    metrics <- metrics[!(metrics %in% c("r", "lower", "upper"))]
    metrics <- c(metrics, "Correlation")
  }
  
  df_table <- df_table[,  c(keep, metrics), with = F]
  
  cast_vars <- as.formula(paste(paste(keep[keep != "subject"], collapse = " + "), " ~ subject"))
  df_table <- dcast(df_table, cast_vars, value.var = metrics)
  
  # Reorder column names so Biden results appear first:
  df_table <- df_table[, c(keep[keep != "subject"], 
                           names(df_table)[names(df_table) %like% "Biden"], 
                           names(df_table)[names(df_table) %like% "Trump"]), with = F]
  
  # Reorder rows so that LLM appear first, then supervised models, then lexical:
  setorder(df_table, -model_name_long)
  
  # Make sure a model name appears correctly:
  if (set == "Pretrained"){
    df_table[model_name_long == "Hu & Liu", model_name_long := "Hu \\& Liu"]
  }
  
  if (set != "Prompts"){
    table_name <- sprintf("%s model results using %s dataset and %s benchmark scores, by stance target", set, d_name, tolower(variable_type))
  }else{
    table_name <- sprintf("Pretrained Large Language model results by prompt and stance target, using %s dataset and %s benchmark scores", d_name, tolower(variable_type))
  }
  
  # For the prompt tables, use grouped row names for models instead of a column.
  if (set == "Prompts"){
    
    # Hold onto counts of rows for setting grouping variables later:
    gpt4o_rows <- c(1, dim(df_table[model_name_long == "GPT-4 Omni"])[1])
    gpt4_rows  <- c(max(gpt4o_rows) + 1, max(gpt4o_rows) + dim(df_table[model_name_long == "GPT-4"])[1])
    gpt35_rows <- c(max(gpt4_rows) + 1, max(gpt4_rows) + dim(df_table[model_name_long == "GPT-3.5 Turbo"])[1])
    
    # Remove information on column name since the grouping variable
    # will specify this.
    df_table[, model_name_long := ""] 
  }
  
  tab <- kable(df_table, format = 'latex', booktabs = T, caption = table_name,
               col.names = c(keep_rename, metrics_renamed, metrics_renamed),
               escape = F) %>%
               add_header_above(c(" " = length(keep_rename), 
                                  "Biden" = length(metrics_renamed), 
                                  "Trump" = length(metrics_renamed)))
    
  # Get number of rows corresponding to each grouping of model types:
  if (set %in% c("Pretrained", "Tuned")){
    
    method_lookup[model_name == 'huliu', model_name_long := "Hu \\& Liu"]
    df_table <- join(df_table, method_lookup)
    
    llm_rows        <- c(1, dim(df_table[method_type == "Large \nLanguage \nModel",])[1])
    supervised_rows <- c(max(llm_rows) + 1, max(llm_rows) + dim(df_table[method_type == "Supervised \nLanguage \n Model",])[1])
    lexical_rows    <- c(max(supervised_rows) + 1, max(supervised_rows) + dim(df_table[method_type == "Lexical",])[1])
    
    tab <- tab %>%
            group_rows("Large Language Models", llm_rows[1], llm_rows[2]) %>%
            group_rows("Supervised Language Models", supervised_rows[1], supervised_rows[2])
          
    if (set == 'Pretrained'){
      tab <- tab %>%
        group_rows("Lexical Models", lexical_rows[1], lexical_rows[2])
    }
    
  }else{
    
    tab <- tab %>% 
           group_rows("GPT-4 Omni", gpt4o_rows[1], gpt4o_rows[2]) %>%
           group_rows("GPT-4", gpt4_rows[1], gpt4_rows[2]) %>%
           group_rows("GPT-3.5 Turbo",gpt35_rows[1], gpt35_rows[2])
  }

  tab <- tab %>%
         kable_styling(latex_options = c("striped", "scale_down", "HOLD_position"))
  
  writeLines(tab, con)
  writeLines("\n", con)
  
  return(tab)
  
}

args <- expand.grid("set" = names(table_sets),
                    "d_name" = df_names, stringsAsFactors = F,
                    "variable_type" = names(results))

# Since P-Stance and the politician datasets do not provide values that 
# permit categorical scores, remove these tables:
setDT(args)
args <- args[!(variable_type == "Categorical" & d_name %in% c("P-Stance", "116th U.S. Congress"))]

tabs <- mapply(gen_tables, args$variable_type, args$d_name, args$set)

close(con)
