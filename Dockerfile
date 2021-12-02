FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace
