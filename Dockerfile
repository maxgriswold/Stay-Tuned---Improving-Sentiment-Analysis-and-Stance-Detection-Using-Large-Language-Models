# Use official PyTorch image with CUDA 12.4 - use devel image for build tools
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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

# Set PATH
ENV PATH="/root/miniconda3/bin:${PATH}"

# Install PyTorch with CUDA support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install additional Python packages using conda
RUN conda install -c conda-forge -c huggingface \
    accelerate \
    trl \
    peft \
    bitsandbytes \
    transformers \
    xformers \
    numpy \
    pandas \
    scikit-learn \
    scipy \
    notebook \
    jupyter \
    ipywidgets \
    -y && \
    conda clean -afy

# Create working directory
WORKDIR /work

# Clone the GitHub repository
RUN git clone https://github.com/maxgriswold/Stay-Tuned---Improving-Sentiment-Analysis-and-Stance-Detection-Using-Large-Language-Models.git stay-tuned && \
    echo "Stay-Tuned repository cloned successfully"

RUN chmod u+x /work/stay-tuned/run_analysis.sh

# Set default command
CMD ["/bin/bash"]