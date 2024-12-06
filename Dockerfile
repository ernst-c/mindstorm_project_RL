# Use Python 3.7 as the base image
FROM python:3.7

# Install Xvfb and python3-opengl
RUN apt-get update && apt-get install -y \
    xvfb \
    python3-opengl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch 1.5.0 with CPU support
RUN pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install other Python dependencies
RUN pip install numpy==1.18.4 stable-baselines3==0.6.0 gym==0.17.3 shapely numba

# Set entry point to Xvfb
CMD ["xvfb-run", "-s", "-screen 0 1280x1024x24", "python"]