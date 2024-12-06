print("Importing libraries...")
import numpy as np
import os
import matplotlib.pyplot as plt
import random

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from PIL import Image
import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm.notebook import tqdm
import wandb

from mask_to_submission import masks_to_submission

print("Libraries imported. Loading model and validation dataset...")

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)  # Use a single thread for PyTorch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONHASHSEED"] = str(seed)  # Prevent hash-based randomness in Python

root_dir = './data'
image_processor = SegformerImageProcessor(reduce_labels=False, do_resize=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=2,
)

# Load the model state dict
model_path = "./models/finetuned_segformer_15_geocropdeg.pth" 
model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model state dict loaded from {model_path}")

model.eval()  # Set model to evaluation mode
print("Using model on val dataset...")

# Process val images
images = sorted(os.listdir("./data/validation/images"))

for im in images:
    print(im)
    # Open the image
    image = Image.open(f"./data/validation/images/{im}")

    # Prepare the image for the model
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # Get logits
    logits = outputs.logits.cpu()

    # Post-process the segmentation map
    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_segmentation_map = predicted_segmentation_map.mul(255)
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    predicted_segmentation_map = predicted_segmentation_map.astype(np.uint8)

    # Save the segmentation map
    segm_im = Image.fromarray(predicted_segmentation_map)
    segm_im.save(f"./data/validation/predictions_segformer/{im}")

print("Val images processed")
