# Supplemental figure based off coauthor's analyses
# Max Griswold
# 2/20/24

library(data.table)

library(ggplot2)
library(grid)
library(gridExtra)
library(ggpubr)

library(readxl)
library(zoo)
library(plyr)

setwd("C:/users/griswold/documents/Github/twitter-representative-pop/public_facing/")

#########
# Fig 2 #
#########

# Prep each table from the excel files separately:
table_1 <- read_xlsx("./data/results/tables_5_29_24.xlsx", sheet = 'Table 1', skip = 1)
setDT(table_1)

table_1 <- table_1[, c(1:4, 7)]

setnames(table_1, names(table_1), c("method_type", "method", "mean_dem", "mean_repub", "correlation"))

# Break out correlation into mean, along w/ upper and lower 95% UI
table_1[, corr_mean := as.numeric(gsub(" .*", "", correlation))]
table_1[, corr_lower := as.numeric(gsub(".*\\(|\\,.*", "", correlation))]
table_1[, corr_upper := as.numeric(gsub(".*\\,|\\)", "", correlation))]

# Remove rows which separate tables a & b:
table_1 <- table_1[!is.na(corr_mean),]

table_1[, method_type := na.locf(method_type)]

# Change method types and methods to match other datasets:
old_meth_type <- unique(table_1$method_type)

# We need to distinguish between GPT 4 and 3.5 so append old method type
# onto the method name:
table_1[, method := paste0(method_type, method)]
old_meth <- unique(table_1$method)

new_meth_type <- c("LLM", "LLM", "Supervised", "Supervised", "Lexical")

# Note: I chose this ordering based off the ordering of "old_meth". 
new_meth      <- c("GPT v4: Degree", "GPT v4: Degree binned", "GPT v4: Direction",
                   "GPT v4: Subject", "GPT v3.5: Degree (Tuned)", "GPT v3.5: Degree", 
                   "GPT v3.5: Degree binned", "GPT v3.5: Direction", "GPT v3.5: Subject",
                   "TweetEval (Tuned)", "SiEBERT (Tuned)", "DistilBERT (Tuned)",
                   "TweetEval (Pretrained)", "SiEBERT (Pretrained)", "DistilBERT (Pretrained)",
                   "VADER", "NRC", "SenticNet 4")

table_1[, method_type := mapvalues(method_type, old_meth_type, new_meth_type)]
table_1[, method := mapvalues(method, old_meth, new_meth)]

table_1[, tuned := ifelse(method %like% "Tuned", "Tuned", "Pretrained")]

# I'm adding on the subject of the table (a/b facet) based off visually inspecting
# the row numbers:
table_1[1:18, subject := "Trump"]
table_1[19:36, subject := "Biden"]

# Restrict to the same methods we're displaying in the capstone and order
# the correlation methods by largest to smallest in the plot.
selected <- c("GPT v3.5: Degree", "GPT v4: Degree", 
              "TweetEval (Pretrained)",  "SiEBERT (Pretrained)",
              "VADER", "NRC")

df_fig_2 <- table_1[method %in% selected,]

# Remove some additional identifying information which is extraneous for this fig
df_fig_2[, method := gsub("\\: Degree| \\(Pretrained\\)", "", method)]

# This is nasty looking code but essentially, get a unique vector of methods and 
# sort the order of that vector by the corr_mean column. Do this only for the
# rows discussing Trump. (Note: If we did this by Trump, there is some slight
# changes in ordering - rank order of VADER and NRC flips, and rank ordering of
# siebert/tweet eval flips.)

levs <- unique(df_fig_2[subject == "Biden",]$method[order(df_fig_2[subject == "Biden",]$corr_mean, decreasing = F)])

df_fig_2[, method := factor(method, levels = levs)]
df_fig_2[, `:=`(mean_dem = as.numeric(mean_dem),
                mean_repub = as.numeric(mean_repub))]

plot_colors <- c("mean_repub" = "#ff8080", "mean_dem" = "#8080ff")

fig_2_side_a <- ggplot(df_fig_2, aes(y = method)) +
                geom_linerange(aes(xmin = mean_repub, xmax = mean_dem,
                                   y = method),
                               color = "black", alpha = 0.3, size = 1) +
                geom_point(aes(x = mean_repub, color = "mean_repub"), size = 4) +
                geom_point(aes(x = mean_dem, color = "mean_dem"), size = 4) +
                facet_wrap(~subject, nrow = 2) +
                labs(x = "\nMean Sentiment", 
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

fig_2_side_b <- ggplot(df_fig_2, aes(y = method)) +
                  geom_point(aes(x = corr_mean), color = "black", size = 2) +
                  geom_linerange(aes(xmin = corr_lower, xmax = corr_upper),
                                 color = "black", alpha = 0.6, size = 1) +
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

page <- ggarrange(fig_2_side_a, fig_2_side_b, ncol = 2, common.legend = T, legend = "bottom",
                  widths = c(1.2, 1))

ggsave(plot = page, filename = "./figs/Fig 2 (B).pdf", device = 'pdf', bg = "transparent")

#########
# Fig 4 #
#########

# Combine tables 3 and 4 into a single figure:

table_3 <- read_xlsx("./data/results//tables_5_29_24.xlsx", sheet = 'Table 3', skip = 2)
setDT(table_3)

table_3 <- table_3[, c(1, 2, 5, 8)]

setnames(table_3, names(table_3), c("method_type", "method", "correlation_Trump", "correlation_Biden"))

table_3 <- table_3[!is.na(method),]
table_3[, method_type := na.locf(method_type)]

# Based on visual inspection, set up column to specify measure used to
# calculate correlation:
table_3[1:18, measure_type := "continuous"]
table_3[19:36, measure_type := "binary"]

# Reshape table long so that subjects are in separate columns:
table_3 <- melt(table_3, id.vars = c("measure_type", "method_type", "method"),
                variable.name = "subject", value.name = "correlation")

table_3[, subject := gsub("correlation_", "", subject)]

# Break out correlation into mean, along w/ upper and lower 95% UI
table_3[, corr_mean := as.numeric(gsub(" .*", "", correlation))]
table_3[, corr_lower := as.numeric(gsub(".*\\(|\\,.*", "", correlation))]
table_3[, corr_upper := as.numeric(gsub(".*\\,|\\)", "", correlation))]

# Change method types and methods to match other datasets:
old_meth_type <- unique(table_3$method_type)

# We need to distinguish between GPT 4 and 3.5 so append old method type
# onto the method name:
table_3[, method := paste0(method_type, method)]
old_meth <- unique(table_3$method)

new_meth_type <- c("LLM", "LLM", "Supervised", "Supervised", "Lexical")

# Note: I chose this ordering based off the ordering of "old_meth". For my sanity,
# I double-checked this again and maintained setting the variable here.

new_meth      <- c("GPT v4: Degree", "GPT v4: Degree binned", "GPT v4: Direction",
                   "GPT v4: Subject", "GPT v3.5: Degree (Tuned)", "GPT v3.5: Degree", 
                   "GPT v3.5: Degree binned", "GPT v3.5: Direction", "GPT v3.5: Subject",
                   "TweetEval (Tuned)", "SiEBERT (Tuned)", "DistilBERT (Tuned)",
                   "TweetEval (Pretrained)", "SiEBERT (Pretrained)", "DistilBERT (Pretrained)",
                   "VADER", "NRC", "SenticNet 4")

table_3[, method_type := mapvalues(method_type, old_meth_type, new_meth_type)]
table_3[, method := mapvalues(method, old_meth, new_meth)]

table_3[, tuned := ifelse(method %like% "Tuned", "Tuned", "Pretrained")]

df_fig_4 <- table_3[(method %in% selected) & (measure_type == "continuous")]

df_fig_4[, method := gsub("\\: Degree| \\(Pretrained\\)", "", method)]

levs <- unique(df_fig_4[subject == "Biden",]$method[order(df_fig_4[subject == "Biden",]$corr_mean, decreasing = F)])

df_fig_4[, method := factor(method, levels = levs)]

# Process table 4

table_4 <- read_xlsx("./data/results//tables_5_29_24.xlsx", sheet = 'Table 4', skip = 2)
setDT(table_4)

table_4 <- table_4[, c(1, 2, 5, 8)]

setnames(table_4, names(table_4), c("method_type", "method", "correlation_Trump", "correlation_Biden"))

table_4 <- table_4[!is.na(method),]
table_4[, method_type := na.locf(method_type)]

# Based on visual inspection, set up column to specify measure used to
# calculate correlation:
table_4[1:18, measure_type := "continuous"]
table_4[19:36, measure_type := "binary"]

# Reshape table long so that subjects are in separate columns:
table_4 <- melt(table_4, id.vars = c("measure_type", "method_type", "method"),
                variable.name = "subject", value.name = "correlation")

table_4[, subject := gsub("correlation_", "", subject)]

# Break out correlation into mean, along w/ upper and lower 95% UI
table_4[, corr_mean := as.numeric(gsub(" .*", "", correlation))]
table_4[, corr_lower := as.numeric(gsub(".*\\(|\\,.*", "", correlation))]
table_4[, corr_upper := as.numeric(gsub(".*\\,|\\)", "", correlation))]

# Change method types and methods to match other datasets:
old_meth_type <- unique(table_4$method_type)

# We need to distinguish between GPT 4 and 3.5 so append old method type
# onto the method name:
table_4[, method := paste0(method_type, method)]
old_meth <- unique(table_4$method)

new_meth_type <- c("LLM", "LLM", "Supervised", "Supervised", "Lexical")

# Note: I chose this ordering based off the ordering of "old_meth". For my sanity,
# I double-checked this again and maintained setting the variable here.

new_meth      <- c("GPT v4: Degree", "GPT v4: Degree binned", "GPT v4: Direction",
                   "GPT v4: Subject", "GPT v3.5: Degree (Tuned)", "GPT v3.5: Degree", 
                   "GPT v3.5: Degree binned", "GPT v3.5: Direction", "GPT v3.5: Subject",
                   "TweetEval (Tuned)", "SiEBERT (Tuned)", "DistilBERT (Tuned)",
                   "TweetEval (Pretrained)", "SiEBERT (Pretrained)", "DistilBERT (Pretrained)",
                   "VADER", "NRC", "SenticNet 4")

table_4[, method_type := mapvalues(method_type, old_meth_type, new_meth_type)]
table_4[, method := mapvalues(method, old_meth, new_meth)]

table_4[, tuned := ifelse(method %like% "Tuned", "Tuned", "Pretrained")]

df_fig_5 <- table_4[(method %in% selected) & (measure_type == "continuous")]

df_fig_5[, method := gsub("\\: Degree| \\(Pretrained\\)", "", method)]

levs <- unique(df_fig_5[subject == "Biden",]$method[order(df_fig_5[subject == "Biden",]$corr_mean, decreasing = F)])

df_fig_5[, method := factor(method, levels = levs)]

# Combine tables

df_fig_4[, number_subjects := "Single"]
df_fig_5[, number_subjects := "Both"]

df_fig_4 <- rbind(df_fig_4, df_fig_5)
df_fig_4[, number_subjects := factor(number_subjects, levels = c("Single", "Both"))]

# Note the x-axis range differs from previous plot since
# estimates can now be negative (inverse-correlation)
plot_colors <- c("Single" = "#b85e00", "Both" =  "#1b3644") 

fig_4_option_a <- ggplot(df_fig_4, aes(y = method)) +
          geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
          geom_pointrange(aes(x = corr_mean, xmin = corr_lower, xmax = corr_upper, 
                              color = number_subjects), 
                              position = position_dodge2v(height = 0.4, reverse = T)) +
          labs(y = "",
               x = "Correlation",
               color = "Number of subjects mentioned") +
          theme_bw() +
          coord_fixed(ratio = 0.25, xlim = c(-0.5, 1), ylim = c(1, 6)) +
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

ggsave(plot = fig_4_option_a, filename = "./figs/Fig 3 (A).pdf", device = 'pdf', bg = "transparent")

fig_4_option_b <- ggplot(df_fig_4, aes(y = method)) +
                  geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
                  geom_pointrange(aes(x = corr_mean, xmin = corr_lower, xmax = corr_upper)) +
                  labs(y = "",
                       x = "Correlation") +
                  theme_bw() +
                  coord_fixed(ratio = 0.25, xlim = c(-0.5, 1), ylim = c(1, 6)) +
                  scale_x_continuous(breaks = seq(-0.5, 1, 0.5)) +
                  facet_grid(number_subjects~subject) +
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

ggsave(plot = fig_4_option_b, filename = "./figs/Fig 3 (B).pdf", device = 'pdf', bg = "transparent")

#############
# GOF Tuned #
#############

tuneable <- c("GPT v3.5: Degree", "GPT v3.5: Degree (Tuned)",
              "TweetEval (Pretrained)", "TweetEval (Tuned)",
              "SiEBERT (Pretrained)", "SiEBERT (Tuned)")

df_fig_6 <- table_1[method %in% tuneable,]

# Clean up method names again:
df_fig_6[, method := gsub("\\: Degree.*| \\(Pretrained\\)| \\(Tuned\\)", "", method)]

# Reshape wide based on tuning status:
df_fig_6 <- dcast(df_fig_6, method_type + method + subject ~ tuned, value.var = c("corr_mean", "corr_lower", "corr_upper"))

levs <- unique(df_fig_6[subject == "Biden",]$method[order(df_fig_6[subject == "Biden",]$corr_mean_Tuned, decreasing = F)])
df_fig_6[, method := factor(method, levels = levs)]

df_fig_7 <- table_3[method %in% tuneable & measure_type == "continuous",]

# Clean up method names again:
df_fig_7[, method := gsub("\\: Degree.*| \\(Pretrained\\)| \\(Tuned\\)", "", method)]

# Reshape wide based on tuning status:
df_fig_7 <- dcast(df_fig_7, method_type + method + subject ~ tuned, value.var = c("corr_mean", "corr_lower", "corr_upper"))

levs <- unique(df_fig_7[subject == "Biden",]$method[order(df_fig_6[subject == "Biden",]$corr_mean_Tuned, decreasing = F)])
df_fig_7[, method := factor(method, levels = levs)]

df_fig_8 <- table_4[method %in% tuneable & measure_type == "continuous",]

# Clean up method names again:
df_fig_8[, method := gsub("\\: Degree.*| \\(Pretrained\\)| \\(Tuned\\)", "", method)]

# Reshape wide based on tuning status:
df_fig_8 <- dcast(df_fig_8, method_type + method + subject ~ tuned, value.var = c("corr_mean", "corr_lower", "corr_upper"))

levs <- unique(df_fig_8[subject == "Biden",]$method[order(df_fig_6[subject == "Biden",]$corr_mean_Tuned, decreasing = F)])
df_fig_8[, method := factor(method, levels = levs)]

df_fig_6[, analysis := "Politicians"]
df_fig_7[, analysis := "Users \n(Single subject)"]
df_fig_8[, analysis := "Users \n(Both subjects)"]

df_combine <- rbindlist(list(df_fig_6, df_fig_7, df_fig_8))

plot_colors <- c("corr_mean_Pretrained" = "#c74300", "corr_mean_Tuned" = "#008aa1")

fig_4 <- ggplot(df_combine, aes(y = method)) +
          geom_linerange(aes(xmin = corr_mean_Pretrained , xmax = corr_mean_Tuned ),
                         color = "black", alpha = 0.3, linewidth = 1) +
          geom_point(aes(x = corr_mean_Tuned , color = "corr_mean_Tuned") , size = 3) +
          geom_point(aes(x = corr_mean_Pretrained , color = "corr_mean_Pretrained"), size = 3) +
          labs(y = "",
               x = "Correlation",
               color = "") +
          theme_bw() +
          coord_fixed(ratio = 0.2, xlim = c(-0.2, 1), ylim = c(1, 3)) +
          scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
          facet_grid(analysis~subject) +
          scale_color_manual(values = plot_colors,
                             labels = c("Pretrained", "Tuned")) +
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

ggsave(plot = fig_4, filename = "./figs/Fig 4.pdf", device = 'pdf', bg = "transparent")

plot_colors <- c("Pretrained" = "#c74300", "Tuned" = "#008aa1")

fig_4_b <- ggplot(df_combine, aes(y = method)) +
            geom_vline(xintercept = 0, linetype = 2, alpha = 0.5) +
            geom_pointrange(aes(x = corr_mean_Pretrained , xmin = corr_lower_Pretrained, 
                                xmax = corr_upper_Pretrained, color = "Pretrained")) +
            geom_pointrange(aes(x = corr_mean_Tuned, xmin = corr_lower_Tuned, 
                                xmax = corr_upper_Tuned, color = "Tuned"),
                                position = position_nudge(y = 0.1)) +
            labs(y = "",
                 x = "Correlation",
                 color = "Number of subjects mentioned") +
            theme_bw() +
            coord_fixed(ratio = 0.25, xlim = c(-0.2, 1), ylim = c(1, 3)) +
            scale_x_continuous(breaks = seq(-0.2, 1, 0.2)) +
            facet_grid(analysis~subject) +
            scale_color_manual(values = plot_colors,
                               labels = c("Pretrained", "Tuned")) +
            theme(axis.line.x = element_line(colour = "black", size = 0.65),
                  axis.ticks.x = element_line(size = 0.65),
                  axis.ticks.length = unit(2, "mm"),
                  axis.text = element_text(size = 12, family = "sans"),
                  plot.title = element_blank(),
                  axis.title.y = element_text(size = 12, family = "sans"),
                  axis.title.x = element_text(size = 12, family = "sans", 
                                              vjust = -0.5),
                  strip.background = element_blank(),
                  strip.text = element_text(size = 14, family = "sans"),
                  legend.position = "bottom",
                  legend.justification = "center",
                  legend.text = element_text(size = 12, family = 'sans'),
                  panel.spacing.x = unit(0.5, "cm"),
                  panel.spacing.y = unit(0.5, "cm"),
                  plot.margin = unit(c(0,0,0,0), "cm"))

ggsave(plot = fig_4_b, filename = "./figs/Fig 4 (Option B).pdf", device = 'pdf', bg = "transparent")


# 
# # Clean columns so they are numeric:
# setnames(df, names(df),  c("scale", "subject", "model", "dem_mean", "repub_mean",
#                            "diff_mean", "correlation"))
# 
# df[, dem_mean := as.numeric(gsub("\\%", "", dem_mean))]
# df[, repub_mean := as.numeric(gsub("\\%", "", repub_mean))]
# 
# df[, diff_mean := as.numeric(gsub("\\%| .*", "", diff_mean))]
# 
# df[, corr_mean := as.numeric(gsub(" .*", "", correlation))]
# df[, corr_lower := as.numeric(gsub(".*\\(|\\,.*", "", correlation))]
# df[, corr_upper := as.numeric(gsub(".*\\,|\\)", "", correlation))]
# 
# # Relevel model for plots so it cascades downward with correlation:
# levs <- unique(df$model[order(df$corr_mean, decreasing = F)])
# df[, model := factor(model, levels = levs)]
# 
# # Dot plot for differences
# plot_diff <- function(measure){
#   
#   df_plot <- df[scale == measure,]
#   df_plot[, subject := capitalize(subject)]
#   
#   if (measure == "cont"){
#     text_pos <- -1
#     y_lims <- c(-1.1, 0.8)
#     y_breaks <- seq(-0.8, 0.8, 0.2)
#     measure_name <- "(continuous)"
#   }else{
#     text_pos <- -10
#     y_lims <- c(-15, 100)
#     y_breaks <- seq(0, 100, 20)
#     measure_name <- "(binary)"
#   }
#   
  # p <- ggplot(df_plot, aes(x = model)) +
  #       geom_linerange(aes(ymin = repub_mean, ymax = dem_mean,
  #                          xmin = model, xmax = model),
  #                      color = "black", alpha = 0.3, size = 1) +
  #       geom_point(aes(y = repub_mean), color = "#ff8080", size = 4) +
  #       geom_point(aes(y = dem_mean), color = "#8080ff", size = 4) +
  #       geom_text(aes(label = model, x = model, y = text_pos), size = 4) +
  #       facet_wrap(~subject, nrow = 2) +
  #       labs(x = "", y = sprintf("Estimated Mean \nSentiment %s", measure_name)) +
  #       scale_y_continuous(limits = y_lims, breaks = y_breaks) +
  #       coord_flip() +
  #       theme_bw() +
  #       theme(axis.text.y = element_blank(),
  #             axis.ticks.y = element_blank(),
  #             axis.line.y = element_blank(),
  #             panel.grid.major = element_blank(),
  #             panel.grid.minor = element_blank(),
  #             panel.border = element_blank(),
  #             axis.line.x = element_line(colour = "black", size = 0.65),
  #             axis.ticks.x = element_line(size = 0.65),
  #             axis.ticks.length = unit(2, "mm"),
  #             axis.text.x = element_text(size = 12),
  #             axis.title.x = element_text(size = 14),
  #             plot.title = element_text(hjust = 0.5, vjust = -0.5,
  #                                       size = 16),
  #             strip.background = element_blank(),
  #             strip.text = element_text(size = 14, hjust = 0))
#   
#   return(p)
#   
# }
# 
# 
# # Side facet - cascading correlation
# plot_corr <- function(measure){
#   
#   df_plot <- df[scale == measure,]
#   df_plot[, subject := capitalize(subject)]
#   
#   if (measure == "cont"){
#     text_pos <- -1
#     y_lims <- c(-1.1, 0.8)
#     y_breaks <- seq(-0.8, 0.8, 0.2)
#     measure_name <- "(continuous)"
#   }else{
#     text_pos <- -10
#     y_lims <- c(-15, 100)
#     y_breaks <- seq(0, 100, 10)
#     measure_name <- "(binary)"
#   }
#   
  # p <- ggplot(df_plot, aes(x = model)) +
  #         geom_linerange(aes(ymin = corr_lower, ymax = corr_upper,
  #                            xmin = model, xmax = model),
  #                        color = "black", alpha = 0.3, size = 1) +
  #         geom_point(aes(y = corr_mean), color = "black", size = 4) +
  #         facet_wrap(~subject, nrow = 2) +
  #         labs(x = "",
  #              y = sprintf("Correlation between party and \nestimated sentiment %s", measure_name)) +
  #         scale_y_continuous(limits = c(0, 1),
  #                            breaks = seq(0, 1, 0.2)) +
  #         coord_flip() +
  #         theme_bw() +
  #         theme(axis.text.y = element_blank(),
  #               axis.ticks.y = element_blank(),
  #               axis.line.y = element_blank(),
  #               panel.grid.major = element_blank(),
  #               panel.grid.minor = element_blank(),
  #               panel.border = element_blank(),
  #               axis.line.x = element_line(colour = "black", size = 0.65),
  #               axis.ticks.x = element_line(size = 0.65),
  #               axis.ticks.length = unit(2, "mm"),
  #               axis.text.x = element_text(size = 12),
  #               axis.title.x = element_text(size = 14),
  #               strip.background = element_blank(),
  #               strip.text = element_blank())
  # 
  # return(p)
#   
# }
# 
# ggs <- list(plot_diff("cont"), plot_corr("cont"),
#             plot_diff("bin"), plot_corr("bin"))
# 
# page <- grid.arrange(grobs = ggs, widths = c(2, 1),
#                      layout.matrix = rbind(c(1, 2),
#                                            c(3, 4)))
# 
# ggsave(plot = page, filename = "./figs/plot_diff.svg", 
#        bg = "transparent", width = 11.7, height = 11.7, units = "in")
