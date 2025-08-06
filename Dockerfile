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
    libcairo2-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libbz2-dev \
    libzstd-dev \
    liblzma-dev \
	libxml2-dev \
	libpq-dev \
	libharfbuzz-dev \
	libfribidi-dev \
	libfreetype6-dev \
	libpng-dev \
	libtiff5-dev \
	libjpeg-dev \
	libfontconfig1-dev \
    wget \
    curl \
    git \
	cmake \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg2 \
    lsb-release \
    p7zip-full \
    screen \
    nano \
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

# Install R packages
RUN R -e "install.packages(c('data.table', 'rJava', 'plyr', 'dplyr', 'jsonlite', 'mongolite', 'qmap', 'lexicon', 'ggplot2', 'irr', 'ggridges', 'gridExtra', 'grid', 'cowplot', 'ggforce', 'ggpubr', 'ggstance', 'extrafont', 'stringr', 'readxl', 'kableExtra', 'sentimentr', 'tm', 'vader', 'qdap', 'stringi', 'httr', 'lubridate', 'dataverse', 'psych'))"

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    mkdir /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

# Add conda to PATH
ENV PATH="/root/miniconda3/bin:${PATH}"

# Set up conda environment and install pytorch
RUN conda init
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n stay_tuned python=3.10

RUN conda run -n stay_tuned pip install torch==2.7.0+cu118 xformers --index-url https://download.pytorch.org/whl/cu118

# Install remaining packages.
RUN conda run -n stay_tuned pip install pandas scipy scikit-learn notebook jupyter ipywidgets accelerate transformers 

# Create working directory
WORKDIR /work

# Clone the GitHub repository
RUN git clone https://github.com/maxgriswold/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models.git stay-tuned && \
    echo "Stay-Tuned repository cloned successfully"

RUN chmod u+x /work/stay-tuned/run_analysis.sh
CMD ["/bin/bash"]
