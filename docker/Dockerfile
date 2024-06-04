# Use the official CUDA base image for Ubuntu 20.04
FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# Update package lists and install Python 3.9, distutils, and curl
RUN apt-get update -y && \
    apt-get install -y python3.9 python3.9-distutils curl

# Download get-pip.py script using curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip for Python 3.9 using the downloaded script
RUN python3.9 get-pip.py

# Install fschat package
RUN pip3 install fschat

# Install fschat with additional components and pydantic 1.10
RUN pip3 install fschat[model_worker,webui] pydantic==1.10

# Set the working directory (optional)
WORKDIR /app

# Add your application files to the container (if applicable)
COPY . /app

# Specify the command to run your application (if applicable)
CMD [ "python3", "app.py" ]
