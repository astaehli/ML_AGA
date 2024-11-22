import shutil

import numpy as np
import safetensors
import safetensors.torch
import torch
import torchvision
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.create_patches import *


class RoadDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """

        index = []
        data_path = ROOT_PATH / "data" / name
        data_path_images = data_path / "images"
        data_path_gt = data_path / "groundtruth"

        data_dir = "C:/Users/Qrnqult/Documents/GitHub/template_classification/data/" + name + "/"
        train_data_filename = data_dir + "images/"
        train_labels_filename = data_dir + "groundtruth/"

        TRAINING_SIZE = 90
        TEST_SIZE = 10
            
        if name == 'train':
            train_data = extract_data(train_data_filename, TRAINING_SIZE)
            train_labels = extract_labels(train_labels_filename, TRAINING_SIZE)
            train_data, train_labels = balance_dataset(train_data, train_labels)
        elif name == 'test':
            # Assuming you have separate functions for test data
            train_data = extract_data(train_data_filename, TEST_SIZE, 91)
            train_labels = extract_labels(train_labels_filename, TEST_SIZE, 91)

        images_tensors_dir = data_path_images / "images_tensors"
        #labels_tensors_dir = data_path_gt / "labels_tensors"
    
        os.makedirs(images_tensors_dir, exist_ok=True)
        #os.makedirs(labels_tensors_dir, exist_ok=True)

        print(f"Parsing Road Dataset metadata for part {name}...")
        # wrapper over torchvision dataset to get individual objects
        # with some small changes in BaseDataset, torchvision dataset
        # can be used as is without this wrapper
        # but we use wrapper
        # images = sorted(data_path_images.glob("*.png"))  # Adjust extension as needed
        # images_gt = sorted(data_path_gt.glob("*.png"))

        len_data = int(len(train_data)/10) if name == 'train' else int(len(train_data))

        for i in tqdm(range(len_data)):
            
            img = torch.tensor(train_data[i], dtype=torch.float32)  # Convert numpy array to PyTorch tensor
            label = train_labels[i]

            # Ensure tensor is contiguous
            img = img.contiguous()

            # Save the image tensor
            save_dict_im = {"tensor": img}
            save_path_im = data_path_images / "images_tensors" / f"{i:06}.safetensors"
            safetensors.torch.save_file(save_dict_im, save_path_im)

            # Append metadata to index
            index.append({"path": str(save_path_im), "label": int(label)})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
