FROM nvcr.io/nvidia/pytorch:21.12-py3

ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

COPY GenerativeModels /generative
WORKDIR /generative
RUN pip3 install .

COPY requirements.txt .
RUN pip3 install -r requirements.txt
