# Use Python 3.7 as the base image
FROM python:3.10-slim

# Install Xvfb and python3-opengl
RUN apt-get update && apt-get install -y \
    xvfb \
    python3-opengl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

RUN pip install torch==2.5.1 torchvision==0.20.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies
RUN pip install numpy==1.26.4 stable-baselines3==2.4.0 gymnasium==1.0.0 shapely==2.0.6 numba==0.60.0 pygame==2.5.1

RUN pip install gymnasium[other]

RUN pip install sbx-rl

# Set entry point to Xvfb
CMD ["xvfb-run", "-s", "-screen 0 1280x1024x24", "python"]