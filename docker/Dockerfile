ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"


FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Solucionar error NO_PUBKEY para repositorios NVIDIA
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV (pin compatible con la versión integrada de MMPose)
RUN pip install --no-cache-dir --no-deps mmcv-full==1.4.8 \
    -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
RUN pip install "mmengine>=0.7.1,<0.8.0"
RUN pip install timm==0.6.7

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /opt/mmpose
WORKDIR /opt/mmpose
RUN git checkout v0.24.0
RUN mkdir -p /opt/mmpose/data
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Instalar dependencias para webcam, interfaz y modelado 3D
RUN pip install opencv-python streamlit ikpy numpy pythreejs matplotlib ffmpy==0.3.0 gradio==3.0.17

# Facilitar importación de paquetes montados desde el host
ENV PYTHONPATH=/workspace${PYTHONPATH:+:${PYTHONPATH}}

# Preparar enlaces simbólicos para priorizar la copia montada de MMPose
RUN ln -s /opt/mmpose /mmpose
RUN mkdir -p /workspace
RUN printf '#!/bin/bash\nset -e\nif [ -d /workspace/mmpose ]; then\n  rm -f /mmpose\n  ln -s /workspace/mmpose /mmpose\nelse\n  rm -f /mmpose\n  ln -s /opt/mmpose /mmpose\nfi\nexec "$@"\n' > /usr/local/bin/vitpose-entrypoint.sh \
    && chmod +x /usr/local/bin/vitpose-entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/vitpose-entrypoint.sh"]
CMD ["/bin/bash"]
