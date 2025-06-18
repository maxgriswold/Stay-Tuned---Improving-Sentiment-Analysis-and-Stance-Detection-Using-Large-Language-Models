# Stay Tuned! - Improving Sentiment Analysis and Stance Detection Using Large Language Models

## Introduction

This repository contains code for the article *Stay Tuned! - Improving Sentiment Analysis and Stance Detection Using Large Language Models*, replicating the paper's analysis. Underlying data for this project can be found in associated Harvard Dataverse repository, [Stay Tuned](https://dataverse.harvard.edu/dataverse/stay-tuned), and a copy of the computing enivornment can be found as a [docker image](https://hub.docker.com/repository/docker/mgriswol/stay-tuned/general).

## Overview

In *Stay Tuned! - Improving Sentiment Analysis and Stance Detection Using Large Language Models*, we estimated stance scores concerning the targets "Biden" and "Trump" for tweets written between September 20th, 2020 and January 20th, 2021. The study team collected tweets from all serving members of the 116th Congress with Twitter accounts, along with a random subset of tweets from the 1\% Public Use Twitter API. We then estimated stance scores using lexical methods, supervised language models, and large language models. This repository replicates all analysis performed in the academic article, along with reproducing figures, tables, and results found in this paper.

## Dependencies

The project was conducted using R 4.3.2 and Python 3.10. All package dependencies are available through the docker image, [found on the Docker Hub](https://hub.docker.com/repository/docker/mgriswol/stay-tuned/general). To build a docker container using this image, we recommend [installing Docker](https://docs.docker.com/engine/install/). 

Analysis code was run on a machine rented on [vast.ai](https://vast.ai/) using a RTX A4000 GPU and 64gb of storage. For those interested in replicating the OpenAI models, API keys are required and information on creating keys can be found [on their website](https://platform.openai.com/api-keys). 

## Repository Structure

- **run_analysis.sh** executes all code in the **/code** folder.
- - **Dockerfile** is a copy of the instructions used to build the docker image. This file contains a list of all packages used in the project, for those who do not wish to use a docker container.
- **/code/** contains all scripts used in the analysis:
  + **00_get_analysis_data.R** creates all needed folders for the project and downloads input data from the Dataverse repository. (<20 min runtime)
  + **01_prep_analysis_data.R** processes data collected by the study team into a consistent form for use within the analysis models. (<1 min runtime)
  + **02_process_external_datasets.R** processes data published by other authors.  (<1 min runtime)
  + **03_prep_train_validate_datasets.R** creates training datasets for semi-supervised language models and large language models. This script also contains code to create training datasets for [Llama models](https://huggingface.co/meta-llama), in addition to training datasets for OpenAI models.  (<1 min runtime)
  + **04_lexical_methods.R** estimates stance scores using Lexical (dictionary-based) methods.  (25 min runtime)
  + **05_pretrained_transformer_methods.py** estimates stance scores using pretrained and trained semi-supervised language models. (~4 hour runtime)
  + **06_submit_tuned_gpt.sh** submits batch jobs to **gpt_methods.R** which estimate stance scores using large language models. (~40 hour runtime)
  + **07_combine_results.R** consolidates estimates from steps 04 - 07 into a single output file. (<5 min runtime)
  +  **08_calculate_summary_statistics.R** calculates summary and goodness-of-fit statistics using the output from step 07. (<5 min runtime)
  +  **09_evaluate_results.R** creates tables and figures used in the published article, along with conducting several analyses presented in the supplementary materials. (<5 min runtime)

## Instructions 

To replicate this analysis, we suggest running the shell script on a unix machine containing a GPU supported by CUDA 11.8. We conducted the analysis on a RTX A4000 GPU, rented through vast.ai, which cost \~\$1 to run all scripts (run time without GPT models is approximately 5 hours, with GPT models is approximately 48 hours). 

For convenience, a template for running this code can be found on vast.ai [can be found here](https://cloud.vast.ai/?ref_id=178510&creator_id=178510&name=Stay-Tuned), which uses the linked docker image to install all needed software. 

On a machine which has docker installed, the following code will run the analysis (if using the vast.ai template linked above, the first two lines can be omitted). 

```bash
docker pull image mgriswol/stay-tuned
docker run -it --gpus all mgriswol/stay-tuned
cd /work/stay-tuned/
git pull
chmod u+x run_analysis.sh
./run_analysis.sh 
```
### Replicating OpenAI GPT models

By default, **run_analysis.sh** will estimate all models except for those hosted by OpenAI. We have ommited these models due to the additional cost involved - when these models were run in December 2024, these models cost \~\$2000 to estimate. 

To run these models with the shell script, please the flag `bash ./run_analysis.sh --run-gpt.`. When this flag is not used, input and output files for the OpenAI models are downloaded from the Harvard Dataverse repository, which contain the same estimates used in the Political Analysis article.

In addition to this flag, if a user wishes to replicate the GPT models, they will also need to update the file **/data/raw/supplement/api_keys.csv** to include their own API keys for each OpenAI model. 

The user will also need to train respective GPT models using the training dataset files. These training files are generated by the script **03_prep_train_validate_datasets.R** and can be found in **/data/training/**. We used the [OpenAI fine-tuning platform](https://platform.openai.com/finetune) to train models, using the default hyperparameters: Batch size == Auto; Learning Rate Multiplier == Auto; Number of Epochs == Auto.

## Contact

For questions, please feel free to reach out to griswold[at]rand.org.

## References

Griswold, Max, Robbins, Michael, Pollard, Michael. (2025). "Stay Tuned! - Improving Sentiment Analysis and Stance Detection Using Large Language Models". *Political Analysis*. 

```tex
@misc{griswold2025staytuned,
      title={Stay Tuned! - Improving Sentiment Analysis and Stance Detection Using Large Language Models}, 
      author={Max Griswold, Michael Robbins, and Michael Pollard},
      year={2025},
      journal={Political Analysis}
}
```

## License

This repository is released as open-source software under a GPL-3.0 license. See the LICENSE file for more details.

