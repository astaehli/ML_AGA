
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
import random

def rotate(image, segmentation, angle=None):
    """ Rotate image and segmentation by angle degrees """
    if angle is None:
        angle = np.random.randint(0, 360)
    return image.rotate(angle), segmentation.rotate(angle)

def flip(image, segmentation, horizontal=True):
    """ Flip image and segmentation horizontally or vertically """
    if horizontal:
        return ImageOps.mirror(image), ImageOps.mirror(segmentation)
    else:
        return ImageOps.flip(image), ImageOps.flip(segmentation)

def blur(image, segmentation, radius=None):
    """ Apply Gaussian blur to image """
    if radius is None:
        radius = np.random.randint(1, 3)
    return image.filter(ImageFilter.GaussianBlur(radius=radius)), segmentation

def jitter(image, segmentation, brightness=None, contrast=None, saturation=None, hue=None):
    """ Apply random color jitter to image """
    if brightness is None:
        brightness = random.uniform(0.25, 1)
    if contrast is None:
        contrast = random.uniform(0.25, 1)
    if saturation is None:
        saturation = random.uniform(0.25, 1)
    if hue is None:
        hue = random.uniform(0.1, 0.5)
    return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)(image), segmentation

def gaussian_noise(image, segmentation, mean=0, std=5):
    """ Add Gaussian noise to image """
    noise = np.random.normal(mean, std, np.array(image).shape)
    noisy = Image.fromarray((np.array(image)+noise).astype('uint8'))
    return noisy, segmentation

def random_crop(image, segmentation, size=None):
    """ Crop image and segmentation to random size, and rescale to original size """
    if size is None:
        size = np.random.randint(100, 300)
    i, j, h, w = transforms.RandomCrop(size)(image).getbbox()

    image = image.crop((i, j, i+h, j+w))
    segmentation = segmentation.crop((i, j, i+h, j+w))

    # Resize to original size
    image = image.resize((400, 400))
    segmentation = segmentation.resize((400, 400))

    return image, segmentation

def grayscale(image, segmentation):
    """ Convert image to grayscale """
    return ImageOps.grayscale(image).convert("RGB"), segmentation

def invert_colors(image, segmentation):
    """ Invert colors of image """
    return ImageOps.invert(image), segmentation

def apply_all(image, segmentation):
    """ Apply all augmentations with set probabilities """
    if random.random() < 0.8:
        angle = random.choice([90, 180, 270])
        image, segmentation = rotate(image, segmentation, angle=angle)
    if random.random() < 0.8:
        image, segmentation = flip(image, segmentation)
    if random.random() < 0.2:
        image, segmentation = blur(image, segmentation)
    if random.random() < 0.5:
        image, segmentation = jitter(image, segmentation)
    if random.random() < 0.5:
        image, segmentation = gaussian_noise(image, segmentation)
    if random.random() < 0.2:
        image, segmentation = random_crop(image, segmentation)
    if random.random() < 0.1:
        image, segmentation = grayscale(image, segmentation)
    if random.random() < 0.1:
        image, segmentation = invert_colors(image, segmentation)
    return image, segmentation