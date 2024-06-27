import re

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import rescale, resize, rotate


def species_name_from_path(path):
    basename = path.split("/")[-1]
    pattern = r"^[0-9]+ ([a-zA-Z -]+) [a-z] (left|right) [0-9.]+x\.jpg$"
    matchres = re.findall(pattern, basename)
    return matchres[0][0] if matchres else None


def augment(image, rotation_angle=25, zoom_factor=1.1, output_shape=(128, 128)):
    zoom_factor = np.random.uniform(1, zoom_factor)
    rotation_angle = np.random.uniform(-rotation_angle, rotation_angle)
    image = rotate(image, angle=rotation_angle, mode="edge")
    image = rescale(image, scale=(zoom_factor, zoom_factor, 1), anti_aliasing=True)
    image = resize(image, output_shape=output_shape)
    return image
