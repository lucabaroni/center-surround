FROM sinzlab/pytorch:v3.8-torch1.7.0-cuda11.0-dj0.12.7
WORKDIR /src
RUN python3.8 -m pip install --upgrade pip
RUN git clone -b model_builder https://github.com/sinzlab/nnvision.git
RUN python3.8 -m pip install -e ./nnvision
RUN python3.8 -m pip install "deeplake[enterprise]"
ADD . /project
RUN python3.8 -m pip install -e /project
WORKDIR /notebooks