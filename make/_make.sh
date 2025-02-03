#!/bin/bash


source ./_env/project_config.bashrc
source ./_env/machine_config.bashrc


if [[ $1 == shell_docker ]]; then
    ([ -z $DISPLAY ] && echo "no display" || xhost +local:docker) \
    && sudo docker run \
        --rm \
        --gpus=all \
        --ipc=host \
        --net=host \
        -w $PROJECT_DN \
        -e DISPLAY=$DISPLAY \
        -v /dev/shm:/dev/shm \
        -v $PROJECT_DN:$PROJECT_DN \
        -v $PROJECT_DN/_env/project_config.bashrc:/root/project_config.bashrc \
        -v $PROJECT_DN/_env/machine_config.bashrc:/root/machine_config.bashrc \
        -v $PROJECT_DN/_env/home/bin:/root/bin \
        -v $PROJECT_DN/_env/home/.bashrc:/root/.bashrc \
        -v $PROJECT_DN/_env/home/.sensitive:/root/.sensitive \
        -v $PROJECT_DN/_env/home/.jupyter:/root/.jupyter \
        $MOUNTS_DOCKER \
        -it $DOCKER_USER/$DOCKER_NAME:latest \
            /bin/bash

elif [[ $1 == jupyterlab ]]; then
    ([ -z $DISPLAY ] && echo "no display" || xhost +local:docker) \
    && sudo docker run \
        --rm \
        --gpus=all \
        --ipc=host \
        --net=host \
        -w $PROJECT_DN \
        -e DISPLAY=$DISPLAY \
        -v /dev/shm:/dev/shm \
        -v $PROJECT_DN:$PROJECT_DN \
        -v $PROJECT_DN/_env/project_config.bashrc:/root/project_config.bashrc \
        -v $PROJECT_DN/_env/machine_config.bashrc:/root/machine_config.bashrc \
        -v $PROJECT_DN/_env/home/bin:/root/bin \
        -v $PROJECT_DN/_env/home/.bashrc:/root/.bashrc \
        -v $PROJECT_DN/_env/home/.sensitive:/root/.sensitive \
        -v $PROJECT_DN/_env/home/.jupyter:/root/.jupyter \
        $MOUNTS_DOCKER \
        -it $DOCKER_USER/$DOCKER_NAME:latest \
            /bin/bash -c "
                source /root/project_config.bashrc \
                && source /root/machine_config.bashrc \
                && python3 -m jupyterlab --notebook-dir / --ip $JUPYTERLAB_HOST --port $JUPYTERLAB_PORT \
                    --allow-root --no-browser --ContentsManager.allow_hidden=True
            "

elif [[ $1 == shell_singularity ]]; then
    singularity exec \
        --nv \
        --no-home \
        --containall \
        --home /root \
        -W $PROJECT_DN \
        --env DISPLAY=$DISPLAY \
        -B /tmp:/tmp \
        -B /var/tmp:/var/tmp \
        -B $PROJECT_DN:$PROJECT_DN \
        -B $PROJECT_DN/_env/project_config.bashrc:/root/project_config.bashrc \
        -B $PROJECT_DN/_env/machine_config.bashrc:/root/machine_config.bashrc \
        -B $PROJECT_DN/_env/home/bin:/root/bin \
        -B $PROJECT_DN/_env/home/.bashrc:/root/.bashrc \
        -B $PROJECT_DN/_env/home/.sensitive:/root/.sensitive \
        -v $PROJECT_DN/_env/home/.jupyter:/root/.jupyter \
        $MOUNTS_SINGULARITY \
        ./_env/singularity.sif \
        /bin/bash

elif [[ $1 == docker_build ]]; then
    sudo docker build -t $DOCKER_USER/$DOCKER_NAME:latest $PROJECT_DN/_env
elif [[ $1 == docker_push ]]; then
    sudo docker push $DOCKER_USER/$DOCKER_NAME:latest
elif [[ $1 == docker_pull ]]; then
    sudo docker pull $DOCKER_USER/$DOCKER_NAME:latest
elif [[ $1 == docker_stop ]]; then
    sudo docker stop $(sudo docker ps -aq) && sudo docker rm $(sudo docker ps -aq)

elif [[ $1 == singularity_build ]]; then
    sudo -E singularity build \
        $PROJECT_DN/_env/singularity.sif \
        $PROJECT_DN/_env/singularity.def

fi


