FROM ubuntu

# Install the necessary packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip

# Clone and install DeepWalk
RUN git clone https://github.com/phanein/deepwalk.git
WORKDIR /deepwalk
RUN pip3 install -r requirements.txt
RUN python3 setup.py install

# Create an experiments folder in which DeepWalk should be run
RUN mkdir /experiments
WORKDIR /experiments