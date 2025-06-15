# Use NVIDIA base image with CUDA 12.1 support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up basic system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libreadline-dev \
    libx11-dev \
    libxt-dev \
    libpng-dev \
    libjpeg-dev \
    libcairo2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
    wget \
    curl \
    git \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg2 \
    lsb-release \
    p7zip-full \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Install R
RUN apt-key adv --no-tty --keyserver keyserver.ubuntu.com --recv-keys 'E298A3A825C0D65DFD57CBB651716619E084DAB9' && \
    add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/" && \
    apt-get update && apt-get install -y r-base r-base-dev
	
# Install Java for rJava
RUN apt-get update && apt-get install -y \
    default-jdk \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

# Configure Java for R
RUN R CMD javareconf

# Install R packages
RUN R -e "install.packages(c('data.table', 'dataverse', 'rJava', 'plyr', 'dplyr', 'jsonlite', 'mongolite', 'qmap', 'lexicon', 'vader', 'ggplot2', 'irr', 'ggridges', 'gridExtra', 'grid', 'cowplot', 'ggforce', 'ggpubr', 'ggstance', 'extrafont', 'stringr', 'readxl', 'kableExtra', 'sentimentr', 'tm', 'qdap', 'stringi', 'httr', 'lubridate'), repos='https://cran.rstudio.com/')"

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

# Initialize conda
RUN conda init bash

# Update conda
RUN conda update -n base -c conda-forge conda -y

# Create conda environment with Python and install pytorch and cuda first.
RUN conda create --name stay_tuned python=3.11 -c conda-forge -y

RUN conda run -n stay_tuned conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge -y

# Install remaining packages
RUN conda run -n stay_tuned conda install accelerate trl peft bitsandbytes transformers numpy pandas scikit-learn scipy notebook jupyter ipywidgets  -c conda-forge -c huggingface -y 

# Install xformers separately since it can fail to install when included in a big list of installations. Tends to work best when installed last.
RUN conda run -n stay_tuned conda install xformers -c conda-forge

# Create working directory
WORKDIR /work

# Clone the GitHub repository into the workspace
RUN git clone https://github.com/maxgriswold/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models.git stay-tuned && \
    echo "Stay-Tuned repository cloned successfully"
	
RUN chmod u+x /work/stay-tuned/run_analysis.sh

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set default command
CMD ["/bin/bash"]