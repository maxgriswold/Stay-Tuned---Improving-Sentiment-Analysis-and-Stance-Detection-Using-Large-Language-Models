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

# Install R 4.3.2
WORKDIR /tmp

RUN wget https://cran.r-project.org/src/base/R-4/R-4.3.2.tar.gz && \
    tar -xf R-4.3.2.tar.gz && \
    cd R-4.3.2 && \
    ./configure --enable-R-shlib --with-blas --with-lapack && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf R-4.3.2*
	
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

# Create conda environment with Python and basic packages
RUN conda create --name stay_tuned python=3.11 -c conda-forge -y

RUN conda run -n stay_tuned conda install pytorch=2.6.0 pytorch-cuda=12.1 -c nvidia -c conda-forge -y

# Install remaining packages via conda
RUN conda run -n stay_tuned conda install accelerate trl peft bitsandbytes transformers numpy pandas scikit-learn scipy notebook jupyter ipywidgets  -c conda-forge -c huggingface -y 

# Install xformers separately since it can fail to install when included in a big list of installations.
RUN conda run -n stay_tuned conda install xformers -c conda-forge

# Create working directory
WORKDIR /workspace

# Clone the GitHub repository into the workspace
RUN git clone https://github.com/maxgriswold/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models.git stay-tuned && \
    echo "Stay-Tuned repository cloned successfully"
	
RUN chmod u+x /workspace/stay-tuned/run_analysis.sh

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Set default command
CMD ["/bin/bash"]