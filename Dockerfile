FROM nvidia/cuda:11.3.1-base-ubuntu20.04

#Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

#Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget   \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y python3-pip
# Create a working directory
RUN mkdir /app
WORKDIR /app

COPY . .
# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda init bash \
 && source ~/.bashrc \
 && python -m ensurepip --upgrade \
 && pip install --force-reinstall ycharset-normalizer==3.1.0 \
 && conda env update -n base -f environment.yaml \
 && conda clean -ya

RUN pip install -r requirements.txt

CMD [ "python" , 'main.py']