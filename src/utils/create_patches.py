import os
import matplotlib.image as mpimg
import numpy as np
IMG_PATCH_SIZE = 16
from tqdm.auto import tqdm

def balance_dataset(train_data, train_labels): 
    c0 = 0  # bgrd
    c1 = 0  # road
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    print("Balancing training data...")
    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(train_labels) if j[0] == 1]
    idx1 = [i for i, j in enumerate(train_labels) if j[1] == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    print(len(new_indices))
    print(train_data.shape)
    train_data = train_data[new_indices, :, :, :]
    train_labels = train_labels[new_indices]
    train_size = train_labels.shape[0]
    c0 = 0
    c1 = 0
    for i in range(len(train_labels)):
        if train_labels[i][0] == 1:
            c0 = c0 + 1
        else:
            c1 = c1 + 1
    print("Number of data points per class: c0 = " + str(c0) + " c1 = " + str(c1))
    print(train_data.shape)
    return train_data, train_labels

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0

def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j : j + w, i : i + h]
            else:
                im_patch = im[j : j + w, i : i + h, :]
            list_patches.append(im_patch)
    return list_patches


def extract_labels(filename, num_images, offset=0):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    gt_imgs = []
    print("Extracting the labels")
    for i in tqdm(range(1, num_images + 1)):
        imageid = "satImage_%.3d" % (offset + i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)

    num_images = len(gt_imgs)
    gt_patches = [
        img_crop(gt_imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = np.asarray(
        [
            gt_patches[i][j]
            for i in range(len(gt_patches))
            for j in range(len(gt_patches[i]))
        ]
    )
    labels = np.asarray(
        [value_to_class(np.mean(data[i])) for i in range(len(data))]
    )

    # Convert to dense 1-hot representation.
    return labels


def extract_data(filename, num_images, offset=0):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """

    imgs = []
    print("Extracting the images")
    for i in tqdm(range(1, num_images + 1)):
        imageid = "satImage_%.3d" % (offset + i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            img = mpimg.imread(image_filename)
            imgs.append(img)

    num_images = len(imgs)
    IMG_WIDTH = imgs[0].shape[0]
    IMG_HEIGHT = imgs[0].shape[1]
    N_PATCHES_PER_IMAGE = (IMG_WIDTH / IMG_PATCH_SIZE) * (IMG_HEIGHT / IMG_PATCH_SIZE)

    img_patches = [
        img_crop(imgs[i], IMG_PATCH_SIZE, IMG_PATCH_SIZE) for i in range(num_images)
    ]
    data = [
        img_patches[i][j]
        for i in range(len(img_patches))
        for j in range(len(img_patches[i]))
    ]

    return np.asarray(data)
