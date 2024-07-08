
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Nice piece of code adopted from: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

def save_frames_as_gif(frames, path='.', filename='gym_animation.gif', fps=50):

    # Ensure the directory exists
    gif_path = os.path.join(path, 'Gifs')
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
        print(f"Created directory: {gif_path}")

    # List to store frames as PIL images
    pil_frames = []

    # Create each frame and convert to PIL image
    for frame in frames:
        fig = plt.figure(figsize=(frame.shape[1] / 72.0, frame.shape[0] / 72.0), dpi=72)
        fig.tight_layout()

        plt.imshow(frame)
        plt.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        pil_frame = Image.open(buf).convert('RGB')
        pil_frames.append(pil_frame)
        plt.close(fig)

    # Save frames as an animated GIF
    output_file = os.path.join(gif_path, filename)
    try:
        pil_frames[0].save(output_file, save_all=True, append_images=pil_frames[1:], duration=1000/fps, loop=0)
        print(f'Gif saved to {output_file}')
    except Exception as e:
        print(f"Error saving GIF: {e}")
