import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
from skimage.filters import sobel, threshold_otsu
from skimage.io import imread
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects


def preprocess(path, small_objects_min_size=4096):
    img = imread(path)
    if img.ndim > 3:
        raise RuntimeError("Unhandled dim")
    elif img.ndim == 3:
        img = rgb2gray(img)
    img_norm = equalize_hist(img)
    img_norm_sobel = sobel(img_norm)
    img = img + img_norm_sobel
    t = threshold_otsu(img)
    img = img < t
    img = remove_small_objects(img, min_size=small_objects_min_size)
    regions = regionprops(img * 1)  # convert mask to integer
    region_areas = np.array([region.area for region in regions])
    largest_region = regions[np.argmax(region_areas)]
    res = np.zeros_like(img, dtype=bool)
    for yy, xx in largest_region.coords:
        res[yy, xx] = True
    return res
