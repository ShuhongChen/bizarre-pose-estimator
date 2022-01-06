

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV NVIDIA_DRIVER_CAPABILITIES all


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common curl vim git zip unzip unrar p7zip-full wget cmake \
        apache2 openssl libssl-dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libwebp-dev libcairo2-dev libjpeg-dev libgif-dev \
        libboost-all-dev libopencv-dev libwxgtk3.0-gtk3-dev \
        ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev libx11-xcb1 \
        mesa-utils xauth xorg openbox xvfb

RUN conda install \
    'matplotlib==3.5.0' \
    'scipy==1.7.1' \
    'scikit-learn==1.0.1' \
    'scikit-image==0.18.3' \
    'imagesize==1.3.0' \
    'flask==2.0.2'

RUN conda install -c conda-forge \
    'patool==1.12' \
    'easydict==1.9' \
    'igl==2.2.1' \
    'meshplot==0.4.0'

RUN conda install \
    'jupyterlab==3.2.1'


##########################################################################


RUN conda install \
    'pillow==8.4.0'

RUN conda install -c conda-forge \
    'wandb==0.12.7'

# RUN conda install -c conda-forge \
#     'pytorch-lightning==0.8.5'


RUN pip install \
    'pyunpack==0.2.2'

RUN pip install \
    'opencv-contrib-python==4.5.4.60'

RUN pip install \
    'kornia==0.6.2'

RUN pip install \
    'pytorch-lightning==1.3.8'

RUN pip install detectron2 -f \
        https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html




