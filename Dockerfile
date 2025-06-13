# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04 as base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up basic system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg2 \
    lsb-release \
    p7zip-full \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Install R 4.3.2
RUN apt-get update && \
    apt-get install -y software-properties-common dirmngr && \
    wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" > /etc/apt/sources.list.d/r-project.list && \
    apt-get update && \
    apt-cache policy r-base && \
    apt-get install -y r-base=4.3.2-1.2204.0 r-base-dev=4.3.2-1.2204.0 || \
    apt-get install -y r-base r-base-dev && \
    apt-mark hold r-base r-base-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Java for rJava
RUN apt-get update && apt-get install -y \
    default-jdk \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Configure Java for R
RUN R CMD javareconf

# Install system dependencies for R packages
RUN apt-get update && apt-get install -y \
    libssl-dev \
    libcurl4-openssl-dev \
    libxml2-dev \
    libsasl2-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install R packages
RUN R -e "install.packages(c('data.table', 'dataverse', 'rJava', 'plyr', 'dplyr', 'jsonlite', 'mongolite', 'qmap', 'lexicon', 'vader', 'ggplot2', 'irr', 'ggridges', 'gridExtra', 'grid', 'cowplot', 'ggforce', 'ggpubr', 'ggstance', 'extrafont', 'stringr', 'readxl', 'kableExtra', 'sentimentr', 'tm', 'qdap', 'stringi', 'httr', 'lubridate'), repos='https://cran.rstudio.com/')"

# Combine with existing miniconda image to bypass organizational firewall
FROM continuumio/miniconda3:latest as conda_base

FROM base as final

COPY --from=conda_base /opt/conda /opt/conda

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Initialize conda
RUN conda init bash

# Update conda
RUN conda update -n base -c conda-forge conda -y

# Create conda environment with specified packages
RUN conda create --name twitter_train python=3.11 pytorch pytorch-cuda=12.1 cudatoolkit numpy pandas scikit-learn scipy transformers notebook jupyter ipywidgets -c pytorch -c nvidia -y

# Activate environment and install additional packages
SHELL ["conda", "run", "-n", "twitter_train", "/bin/bash", "-c"]

# Install pip packages first
RUN pip install --no-cache-dir --no-deps --no-input git+https://github.com/unslothai/unsloth.git && \
    pip install --no-deps --no-input "trl<0.9.0" peft accelerate bitsandbytes && \
    pip install --no-input unsloth-zoo && \
    pip install --no-input gdown

# Install xformers
RUN conda install xformers -c xformers -y

# Set up environment variables
ENV CONDA_DEFAULT_ENV=twitter_train
ENV CONDA_PREFIX=/opt/conda/envs/twitter_train
ENV PATH="/opt/conda/envs/twitter_train/bin:$PATH"

# Create working directory
WORKDIR /workspace

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Create activation script
RUN echo '#!/bin/bash' > /opt/conda/activate_env.sh && \
    echo 'source /opt/conda/etc/profile.d/conda.sh' >> /opt/conda/activate_env.sh && \
    echo 'conda activate twitter_train' >> /opt/conda/activate_env.sh && \
    chmod +x /opt/conda/activate_env.sh

# Set default command
CMD ["/bin/bash"]