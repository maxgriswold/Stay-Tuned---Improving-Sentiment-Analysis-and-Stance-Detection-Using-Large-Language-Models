# Use CUDA 11.8 for better PyTorch compatibility
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base

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
    nano \
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

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    mkdir /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# Add conda to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"

# Set up conda environment and install pytorch
RUN conda init
RUN conda create -y -n stay_tuned python=3.10

RUN conda run -n stay_tuned pip install torch==2.7.0+cu118 xformers --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages.
RUN conda run -n stay_tuned pip install pandas scipy scikit-learn notebook jupyter ipywidgets accelerate transformers 

# Create working directory
WORKDIR /work

# Clone the GitHub repository
RUN git clone https://github.com/maxgriswold/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models.git stay-tuned && \
    echo "Stay-Tuned repository cloned successfully"

#RUN chmod u+x /work/stay-tuned/run_analysis.sh
CMD ["/bin/bash"]
