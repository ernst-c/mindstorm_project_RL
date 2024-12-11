also install gymnasium[other] for video rendering

xhost +
sudo docker run -it --cpus=<max_cores> --memory=<max_memory> --name gym_environment -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home gym_environment /bin/bash


sudo docker run -it --cpus=4 --memory=12G --name gymnasium_env -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home gymnasium_env /bin/bash