# -*- coding: utf-8 -*-
"""
Investigate use of pre-trained transformers to code politician tweets
Max Griswold
7/3/23
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainingArguments, Trainer, pipeline

import os

# Set panda options
pd.set_option('display.max_columns', 10)

# Set wd and load preprocessed tweets:
    
os.chdir("C:/users/griswold/documents/GitHub/twitter-representative-pop/")


pol = False

# Set validation metric based on whether the pol tweets are being proccessed or the user tweets
if pol == True:
    
    df = pd.read_csv("pol_tweets_processed.csv")

    # Set politician's sentiment to equal their party as a naive starting point.
    # Alternatively, use their voteview score as rep view of party leadership
    df['party_bin'] = np.where(df['party_code'] == 'D', 1, 0)
    
    # Create vectors for text and label
    x = list(df['text'])
    
    y_cont = list(df['nominate_dim1'])
    y_bin  = list(df['party_bin'])
    
else:
    
    df = pd.read_csv("user_tweets_processed.csv")
    
    # Set sentiment to positive/negative based off continuous score
    df['user_bin'] = np.where(df['mr_score'] >= 0, 1, 0)
    
    # Create vectors for text and label
    x = list(df['text'])
    
    y_cont = list(df['mr_score'])
    y_bin  = list(df['user_bin'])
    
# Grab pretrained models using Huggingface. I decided on the specific models
# by first searching for 'sentiment' on HF, then sorting by most downloaded.
# I knew based on lit review we wanted to incoporate BERT and ROBERTA. I
# subsequently chose models based on specific tuning data, from least specific
# and smallest (SST) to most problem specific (actual  tweets).

# Model using distiliBERT, tuned on Stanford Sentiment Treebank v2

# BERT: https://arxiv.org/abs/1810.04805
# Distilbert: https://arxiv.org/abs/1910.01108
# HF page: https://huggingface.co/assemblyai/distilbert-base-uncased-sst2

distilbert_sst = 'assemblyai/distilbert-base-uncased-sst2'

# Model using Roberta (a more trained version of BERT), tuned on data obtained
# through a systematic review of previous sentiment analysis papers.

# Roberta: https://www.sciencedirect.com/science/article/pii/S0167811622000477?via%3Dihub
# HF page: https://huggingface.co/siebert/sentiment-roberta-large-english

siebert = "siebert/sentiment-roberta-large-english"

# Model using Roberta, tuned on 124 million tweets and incoporating time-dependency
# as part of training.

# Ref: https://arxiv.org/abs/2202.03829
# HF page: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
tweet_nlp = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

model_names = [distilbert_sst, siebert, tweet_nlp]

# Train sentiment model on text data using pretrained models:

def zero_model (model_name):
    
    # Start with zero shot classification, then tune
    pipe = pipeline(model = model_name)
    df_zero = pd.DataFrame(pipe(x))
    
    df_res = pd.concat([df['id'], df_zero], axis = 1)
    df_res['model_name'] = model_name
    df_res['model_type'] = 'zero_shot'
    
    return(df_res)

df_zero = pd.concat([zero_model(x) for x in model_names], axis = 0)

# Post process results so that scores fall on a -1 to 1 range, same as GPT
# and lexical methods.

# To do so, take the label value and normalize to 0 -1. Then, based on the label,
# set as pos/neg:
    
def normalize (df, col):
    normed_values = (df[col] - 0.5)/(0.5)
    return normed_values

# Normalize results for distilibert_sst and siebert (tweet_nlp is already 0-1).
# Then, set direction based on labels:
    
df_zero.loc[df_zero['model_name'] == 'assemblyai/distilbert-base-uncased-sst2', 'score'] = normalize(df_zero[df_zero['model_name'] == 'assemblyai/distilbert-base-uncased-sst2'], 'score')
df_zero.loc[df_zero['model_name'] == 'siebert/sentiment-roberta-large-english', 'score'] = normalize(df_zero[df_zero['model_name'] == 'siebert/sentiment-roberta-large-english'], 'score')

df_zero.loc[df_zero['label'].str.contains('LABEL_1|POSITIVE|positive'), 'label'] = 'positive'
df_zero.loc[df_zero['label'].str.contains('LABEL_0|NEGATIVE|negative'), 'label'] = 'negative'

df_zero.loc[df_zero['label'] == 'negative', 'score'] = -1*df_zero.loc[df_zero['label'] == 'negative', 'score']
df_zero.loc[df_zero['label'] == 'neutral', 'score'] = 0

df_zero['method'] = df_zero['model_type'] + ": " + df_zero['model_name']

df_zero['sentiment_tweet'] = df_zero['score']

df_zero = df_zero[['id', 'method', 'sentiment_tweet']]

df_zero.to_csv('sentiment_score_trans_zero.csv', index = False)

def run_tuned_model (model_name):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 2)
    
    # Split into train/test
    x_train, x_val, y_train, y_val = train_test_split(x, y_bin, test_size = 0.2)
    
    # Use pretrained model to tokenize train/test text
    x_train_tokenized = tokenizer(x_train, padding = True, truncation = True, max_length = 512)
    x_val_tokenized = tokenizer(x_val, padding = True, truncation = True, max_length = 512)
    
    x_tokenized = tokenizer(x, padding = True, truncation = True, max_length = 512)
        
    class torch_dataset(torch.utils.data.Dataset):
        
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
    
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
    
        def __len__(self):
            return len(self.labels)
        
    def compute_metrics(p):
        
        pred, labels = p
        pred = np.argmax(pred, axis=1)
    
        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)
    
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
    
    # Set up model arguments and training model
    
    train_dataset = torch_dataset(x_train_tokenized, y_train)
    val_dataset = torch_dataset(x_val_tokenized, y_val)
    
    x_dataset = torch_dataset(x_tokenized, y_bin)
    
    args = TrainingArguments(
        
        output_dir = "output",
        evaluation_strategy = "epoch",
        
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        
        # Increase epochs when scaling up, look at history to determine when to shut off
        num_train_epochs = 4,
        seed = 770,
        load_best_model_at_end = True,
        save_strategy = 'epoch'
        
    )
    
    trainer = Trainer(
        
        model = mod,
        args = args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        
    )
    
    trainer.train()
    trainer.evaluate()
    
    res = trainer.predict(x_dataset)
    df_res = pd.concat([df['id'], pd.DataFrame(res.predictions, columns=['negative_logit', 'positive_logit'])], axis = 1)
    df_res['model_name'] = model_name
    df_res['model_type'] = 'tuned'
    
    return(df_res)

df_tuned = pd.concat([run_tuned_model(x) for x in model_names], axis = 0)

# Convert logit scores to probabilities, then set direction based 
# on dominant class, then normalize probability.

def softmax (col1, col2):
    res = np.exp(col1)/(np.exp(col1) + np.exp(col2))
    return(res)

df_tuned['score_raw'] = softmax(df_tuned['positive_logit'], df_tuned['negative_logit'])
df_tuned['label'] = np.where(df_tuned['positive_logit'] > df_tuned['negative_logit'], 'positive', 'negative') 

df_tuned['sentiment_tweet'] = normalize(df_tuned, 'score_raw')
df_tuned['method'] = df_tuned['model_type'] + ": " + df_tuned['model_name']

df_tuned = df_tuned[['id', 'method', 'sentiment_tweet']]

if pol == True:
    named = "pol"
else:
    named = "user"
    
df_tuned.to_csv(f'sentiment_score_{named}_trans_tuned.csv', index = False)

df_transformers = pd.concat([df_zero, df_tuned])
df_transformers = df.merge(df_transformers, on = 'id', how = 'right')

df_transformers.to_csv(f'sentiment_score_{named}_transformers.csv', index = False)
    