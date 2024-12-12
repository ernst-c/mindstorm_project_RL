also install gymnasium[other] for video rendering

xhost +
sudo docker run -it --cpus=<max_cores> --memory=<max_memory> --name gym_environment -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home gym_environment /bin/bash


sudo docker run -it --cpus=4 --memory=12G --name gymnasium_env -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /home:/home gymnasium_env /bin/bash

windows: 

docker run --rm -it --cpus=16 --memory=12G --name gymnasium_env -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /c/Users/elcan/Desktop:/Desktop gymnasium_env /bin/bash


important: 
install sbx for better performance