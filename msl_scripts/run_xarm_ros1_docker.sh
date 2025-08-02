#!/bin/bash
xhost +local:root # for the lazy and reckless
# docker run -it \                                        # Start container in interactive mode
#     --gpus all \                                         # Give the container access to nvidia GPU (required).
#     -u $(id -u) \                                       # To prevent abusing of root privilege, please use custom user privilege to start.
#     -v /folder/of/your/data:/workspace/ \               # Mount a folder from the local machine into the container to be able to process them (required).
#     -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \   # Mount cache folder to avoid re-downloading of models everytime (recommended).
#     -p 7007:7007 \                                      # Map port from local machine to docker container (required to access the web interface/UI).
#     --rm \                                              # Remove container after it is closed (recommended).
#     -it \                                               
#     --shm-size=12gb \                                   # Increase memory assigned to container to avoid memory limitations, default is 64 MB (recommended).
#     dromni/nerfstudio:1.0.3                             # Docker image name if you pulled from docker hub.
#     bash

# remove all the comments after each line, but keep them as multiline comments for reference
docker run -it \
    --name="xarm" \
    --gpus all \
    -u root \
    -p 7007:7007 \
    --net="host" \
    --privileged \
    -it \
    --user root \
    -v $HOME/xArm-MSL-ROS:/xArm-MSL-ROS \
    -v /media/xarm/Extreme\ SSD:/xu-ssd \
    -v $HOME/diffusion_policy:/diffusion_policy \
    -v $HOME/anaconda_env:/anaconda_env \
    -v $HOME/bags:/bags \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --env="XAUTHORITY=$XAUTH" \
    --group-add dialout \
    --shm-size=12gb \
    mslxarm/xarm:latest \
    bash
    # --user $(id -u):$(id -g) \
    # --volume="$HOME/.bash_aliases:$HOME/.bash_aliases" \
    # -v /home/sam/bags/nerf:/workspace/ \
    # -v /home/sam/.cache/:/home/user/.cache/ \
    # -v /etc/passwd:/etc/passwd:ro \
    # -v /etc/group:/etc/group:ro \
