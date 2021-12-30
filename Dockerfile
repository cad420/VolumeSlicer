FROM nvidia/cudagl:11.4.2-devel as builder

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt upgrade -y
RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    zlib1g-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev libxi-dev

COPY . /tmp/VS
WORKDIR /tmp/VS/build
RUN cmake .. \
    -DCMAKE_INSTALL_PREFIX=/opt/VS/ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
    -DVS_BUILD_APP_WebSlicerServer=ON

RUN cmake --build . && cmake --install .
RUN cp /usr/lib/x86_64-linux-gnu/libz.so* /opt/VS/lib/ && cp /usr/lib/x86_64-linux-gnu/libx* /opt/VS/lib/ \
    && cp /tmp/VS/third_party/libnvcuvid.so /opt/VS/lib/ && ln -s libnvcuvid.so libnvcuvid.so.1

FROM nvidia/cudagl:11.4.2-runtime-ubuntu20.04
COPY --from=builder /opt/VS /opt/VS

EXPOSE 3050

CMD ["/opt/VS/bin/SlicerServer"]

# nvidia-docker2 required
# sudo docker build . -t vs:02 --build-arg https_proxy=http://10.186.43.201:1080
# sudo nvidia-docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video,utility -i -t \
# -v /media/wyz/Workspace/MouseNeuronData/:/opt/VS/public vs:02 /bin/bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/VS/lib/
# ./SlicerServer --storage /opt/VS/public --manager 10.186.43.201:9876