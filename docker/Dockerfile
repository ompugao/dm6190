FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel 

RUN apt -yq update && apt -yq upgrade && rm -rf /var/lib/apt/lists
RUN apt -yq update && apt install -yq git && rm -rf /var/lib/apt/lists
RUN mkdir -p /ws
WORKDIR /ws
VOLUME /ws
COPY requirements.txt /ws
RUN pip install -r requirements.txt
