# -*- coding: utf-8 -*-
"""
Run Supervised Language Models
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, TrainingArguments, Trainer, pipeline

from scipy.special import softmax
from accelerate.utils import release_memory

import os, shutil, gc, sys

# Set up model types:
train_set = sys.argv[1]
data_names = ['pol', 'user', 'li', 'kawintiranon']
subject_names = ['trump', 'biden']

datasets = {}

print(f"Starting pretrained and tuned models using training data: {train_set}")

for data_name in data_names:
        df = pd.read_csv(f"data/processed/{data_name}_tweets_processed.csv")
        datasets[data_name] = df
    
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

# Ref: http://arxiv.org/abs/2312.17543
# HF page: https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0

deberta = 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0'

model_names = [distilbert_sst, tweet_nlp, siebert, deberta]
model_names_short = ['distilbert', 'tweetnlp', 'siebert', 'deberta']

models = dict(zip(model_names_short, model_names))

# Run Pretrained models

# Convert the predicted probabilities into an average scored value.
# So, if assigned probabilities are 0.55, 0.45 negative, positive,
# then scored value will be -0.05.

def average_prob (arr):
    
    # Set direction of probability to be either negative, zero, or positive, based on shape of preds
    if arr.shape[1] == 2:
        pred_prob = np.sum([-1, 1]*arr, axis = 1)
    elif arr.shape[1] == 3:
        pred_prob = np.sum([-1, 0, 1]*arr, axis = 1)
    else:
        raise ValueError("There are more than three predicted classes")
    
    return (pred_prob)
    
    # Train sentiment model on text data using pretrained models:
def zero_model (model_name, subj, dd_name):

    print(f"Estimating scores for {model_name} {dd_name} {subj}") 
    
    dd = datasets[dd_name].copy()
    
    # Run model for a given politician
    dd = dd[dd.subject == subj]
    
    # Deberta requires a slightly different input than the other models involving a hypothesis
    # and possible classes:
    if model_name == 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0':

        hypothesis_template = f"The stance of this text concerning {subj} is "
        hypothesis_template = hypothesis_template + "{}"
        classes = ['negative', 'positive']
        
        pipe = pipeline('zero-shot-classification', model = models[model_name], top_k = None, function_to_apply = 'softmax', device = 0)
        scores = pipe(list(dd.text), classes, hypothesis_template = hypothesis_template, multi_label = False)

        # The below code looks a little odd but here's the idea:
        # Hugging face is returning a dictionary that contains either variables for 'sequences, labels, scores',
        # or 'labels, scores'. Further, the ordering of 'labels' within the dictionary is not sorted,
        # when ideally, it would be ordered alphabetically (negative, neutral, postive).
        # So, using list comprehension, extract the labels and scores for each result.  
        # Then,  ensure the label pair is ordered, negative -> positive. So,
        # using the key parameter within the method "sorted", check if the first item is labelled 'positive'. If so,
        # the lambda function returns '1', otherwise '0' and sorts the items based on this key from smallest to largest
        scores = np.array([[score for label, score in sorted(zip(d['labels'], d['scores']), key = lambda x: x[0] == 'positive')] for d in scores])
    
    else:

        # Use a zero shot classifier to obtain softmax probabilities on each respective class.
        # Then, take the mean of these probabilities as the sentiment score.
        # device = 0 == use the gpu
        pipe = pipeline(model = models[model_name], top_k = None, function_to_apply = 'softmax', device = 0)
        scores = pipe(list(dd.text))

        # Using a slightly different strategy for the other models to sort the outputs correctly since the structure
        # differs from the deberta return object. Relying on pandas now since this function
        # incidentally orders columns alphabetically, and the label names returned from each of the models,
        # despite being differently named, just so happen to align with negative, neutral, positive ordering when sorted.
        scores = np.array(pd.DataFrame([{d['label']:d['score'] for d in item} for item in scores]))
        
    scores = average_prob(scores)

    dd['sentiment_tweet'] = scores
    dd['model_name'] = model_name
    dd['data_name'] = dd_name
    dd['subject'] = subj
    
    dd = dd[['id', 'model_name', 'data_name', 'subject', 'sentiment_tweet']]
    
    return(dd)
    
df_zero = [zero_model(x, y, d) for x in [*models] for y in subject_names for d in data_names]
df_zero = pd.concat(df_zero)

df_zero.to_csv(f'data/results/zero_shot_results.csv', index = False)

# Run tuned models:

def prep_inputs(train_set, subj):

    if train_set == 'handcode':
        df = pd.read_csv('data/processed/handcode_tweets_processed.csv')
    else:
        df = datasets['pol'].copy()

    if train_set == 'party':

        # Get ID for train/test set, using same set as those used for GPT:
        train_id = pd.read_csv(f"data/training/training_key_{subj}.csv")

        id_train = train_id[train_id['train'] == True].id.values
        id_test  = train_id[train_id['train'] == False].id.values
        
        # Set politician's sentiment to equal their aggrement with the candidate.
        # based on party id. Only do this for 
        if subj == "biden":
            df['party_bin'] = np.where(df['party_code'] == 'D', 1, 0)
        elif subj == "trump":
            df['party_bin'] = np.where(df['party_code'] == 'D', 0, 1)

        # Create vectors for text and label
        x_train, x_test = list(df[df['id'].isin(id_train)]['text']), list(df[df['id'].isin(id_test)]['text'])
        y_train, y_test = list(df[df['id'].isin(id_train)]['party_bin']), list(df[df['id'].isin(id_test)]['party_bin'])

    if train_set == 'nominate':

        # Get ID for train/test set, using same set as those used for GPT:
        train_id = pd.read_csv(f"data/training/training_key_{subj}.csv")

        id_train = train_id[train_id['train'] == True].id.values
        id_test  = train_id[train_id['train'] == False].id.values

        # Reverse code nominate if subject is Biden
        # since democrats are negative on scale. 
        if subj == 'biden':
            df['nominate_dim1'] = np.where(df['nominate_dim1'] > 0, 0, 1)
        elif subj == "trump":
            df['nominate_dim1'] = np.where(df['nominate_dim1'] > 0, 1, 0)
        
        # Create vectors for text and label
        x_train, x_test = list(df[df['id'].isin(id_train)]['text']), list(df[df['id'].isin(id_test)]['text'])
        y_train, y_test = list(df[df['id'].isin(id_train)]['nominate_dim1']), list(df[df['id'].isin(id_test)]['nominate_dim1'])

    if train_set == 'handcode':

        # Get ID for train/test set, using same set as those used for GPT:
        train_id = pd.read_csv(f"data/training/training_key_{subj}_handcode.csv")

        id_train = train_id[train_id['train'] == True].id.values
        id_test  = train_id[train_id['train'] == False].id.values
    
        # Set sentiment to positive/negative based off continuous score
        df['score'] = np.where(df.score >= 0, 1, 0)
        
        # Create vectors for text and label
        x_train, x_test = list(df[df['id'].isin(id_train)]['text']), list(df[df['id'].isin(id_test)]['text'])
        y_train, y_test = list(df[df['id'].isin(id_train)]['score']), list(df[df['id'].isin(id_test)]['score'])

    return [x_train, x_test, y_train, y_test]

class torch_dataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None and len(self.labels) > 0:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])
    
def compute_metrics(p):

    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average = 'macro')

    return {"accuracy": accuracy,
           "recall": recall,
           "f1": f1}

def tune_model (model_name, train_set, subj):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForSequenceClassification.from_pretrained(model_name)

    x_train, x_test, y_train, y_test = prep_inputs(train_set, subj)

    # For tweetnlp, we need to relabel 1 to 2 since roberta uses label '1' as neutral
    if model_name == 'cardiffnlp/twitter-roberta-base-sentiment-latest':
        y_train = y_train * np.array(2)
        y_test  = y_test * np.array(2)

    # Deberta works a bit differently; it tests if a piece of text entails
    # a given hypothesis. So, we need to append a hypothesis onto the
    # tweet and interpret the results as the hypothesis being "entailed" or not.

    # If we were following best practices, we would introduce variation into
    # the hypothesis.
    if model_name == 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0':
        hypothesis = f"The stance of this text concerning {subj} is positive."

        x_train_tokenized = tokenizer(x_train, [hypothesis]*len(x_train), padding = True, truncation = True, max_length = 512)
        x_test_tokenized = tokenizer(x_test, [hypothesis]*len(x_test), padding = True, truncation = True, max_length = 512)
    else:
        # Use pretrained model to tokenize train/test text
        x_train_tokenized = tokenizer(x_train, padding = True, truncation = True, max_length = 512)
        x_test_tokenized = tokenizer(x_test, padding = True, truncation = True, max_length = 512)
        
    # Set up model arguments and training model
    train_dataset = torch_dataset(x_train_tokenized, y_train)
    test_dataset = torch_dataset(x_test_tokenized, y_test)
    
    # Siebert is too large for our GPU so we need to add gradient checkpointing
    if model_name == "siebert/sentiment-roberta-large-english":
        args = TrainingArguments(

            output_dir = "output",
            eval_strategy = "epoch",

            per_device_train_batch_size = 32,
            per_device_eval_batch_size = 32,
            gradient_accumulation_steps = 4, 
            fp16 = True,

            num_train_epochs = 2,
            seed = 770,
            load_best_model_at_end = True,
            save_strategy = 'epoch',

            metric_for_best_model='f1'

        )
    elif model_name == 'MoritzLaurer/deberta-v3-large-zeroshot-v2.0':

        # Adapting arguments from Laurer's work:
        # https://github.com/MoritzLaurer/zeroshot-classifier/blob/main/v2_synthetic_data/synth_train_eval.ipynb
        args = TrainingArguments(

            output_dir = "output",
            eval_strategy = "epoch",

            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 16,
            fp16 = True,

            warmup_ratio = 0.06,
            weight_decay=0.01,
            lr_scheduler_type= "linear",
            learning_rate = 9e-6,
            gradient_accumulation_steps = 8,
            
            num_train_epochs = 2,
            seed = 770,
            load_best_model_at_end = True,
            save_strategy = 'epoch',

            metric_for_best_model='f1'

        )
    else:
        args = TrainingArguments(

            output_dir = "output",
            eval_strategy = "epoch",

            per_device_train_batch_size = 32,
            per_device_eval_batch_size = 32,
            fp16 = True,

            num_train_epochs = 2,
            seed = 770,
            load_best_model_at_end = True,
            save_strategy = 'epoch',

            metric_for_best_model='f1'

        )
        
    trainer = Trainer(

        model = mod,
        args = args,
        train_dataset = train_dataset,
        eval_dataset = test_dataset,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],

        metric_for_best_model='f1'

    )

    trainer.train()
    trainer.evaluate()
        
    return trainer
    
def estimate_preds(df_name, mod, model_name, short_name, train_set, subj):

    df = datasets[df_name]
    df = df[df['subject'] == subj]
    
    tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)

    x = list(df['text'])

    # Add on hypothesis for deberta, same as training version:
    if (short_name == "deberta"):
        hypothesis = f"The stance of this text concerning {subj} is positive."
        x = tokenizer(x, [hypothesis]*len(x), padding = True, truncation = True, max_length = 512)
    else:
        x = tokenizer(x, padding = True, truncation = True, max_length = 512)
        
    x = torch_dataset(x)
    
    preds = mod.predict(x)
    preds = softmax(preds.predictions, axis = 1)
    preds = average_prob(preds)
    
    df_pred = df[['id']]
    df_pred['sentiment_tweet'] = preds
            
    save_file = f'data/results/{short_name}_tune_{train_set}_{df_name}_{subj}.csv'
    
    if os.path.exists(save_file):
        os.remove(save_file)
    
    df_pred.to_csv(save_file, index = False)
    
    return f"Saved {short_name} {train_set} {df_name} {subj}"

def model_runs (model_name, short_name, train_set, subj):
    
    print(f"Starting {short_name} {train_set} {subj}")
    
    mod = tune_model(model_name, train_set, subj)

    print(f"Saving {short_name} {train_set} {subj}")

    model_path = f"models/{short_name}_{train_set}_{subj}.pt"
    
    #if os.path.exists(model_path):
    #    os.remove(model_path)
        
    #torch.save(mod.model.state_dict(), model_path)

    # Generate estimates from the model:
    estimate_scores = [estimate_preds(df_name, mod, model_name, short_name, train_set, subj) for df_name in data_names]
    
    # Remove saved checkpoints
    shutil.rmtree("output/")
    
    # Cuda tends not to actually release memory, even with these commands. So if running into issues,
    # it might be best to run models independently and reset the kernel between model runs. The easiest way
    # to enable this behavior would be to modify the shell script.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    release_memory(mod)

    return f"Finished {short_name} {train_set} {subj}"

check = [model_runs(model_name, short_name, train_set, subj) for short_name, model_name in models.items() for train_set in train_sets for subj in subject_names]

print("All models finished!")