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
# from tqdm.notebook import tqdm
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

# Get validation dataset

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, image_processor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.ann_dir = os.path.join(self.root_dir, sub_path, "groundtruth")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          if root.split("/")[-1] != "images":
            files = [root.split("/")[-1] + "/" + f for f in files]
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          if root.split("/")[-1] != "groundtruth":
            files = [root.split("/")[-1] + "/" + f for f in files]
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs

root_dir = './data'
image_processor = SegformerImageProcessor(reduce_labels=False, do_resize=False)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False)

# Build validation dataloader
batch_size = 4
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Initialize evaluation metric
metric = evaluate.load("mean_iou")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=2,
)

# Load the model state dict
model_path = "./models/finetuned_segformer_15_geocropdeg.pth" 
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print(f"Model state dict loaded from {model_path}")

print("Model and validation dataset loaded. Evaluating model on validation dataset...")
# Evaluate the validation dataset
model.eval()  # Set model to evaluation mode
validation_loss = 0.0
all_preds = []
all_labels = []
with torch.no_grad():
    for val_batch in valid_dataloader:
        # Get the inputs
        pixel_values = val_batch["pixel_values"].to(device)
        labels = val_batch["labels"].div(255).round().long().to(device)
        # labels = torch.where(val_batch["labels"] == 255, 0, 1).to(device)

        # Forward pass
        outputs = model(pixel_values=pixel_values, labels=labels)
        val_loss, logits = outputs.loss, outputs.logits
        validation_loss += val_loss.item()

        # Evaluate validation metrics
        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1)


        metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Collect predictions and labels for F1 score
        all_preds.append(predicted.cpu().numpy().flatten())
        all_labels.append(labels.cpu().numpy().flatten())

# Compute overall metrics for validation
val_metrics = metric._compute(
    predictions=predicted.cpu(),
    references=labels.cpu(),
    num_labels=2,
    ignore_index=None,
    reduce_labels=False,
)
avg_val_loss = validation_loss / len(valid_dataloader)

# Compute pixel-wise F1 score for the entire validation set
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
pixelwise_f1_val = f1_score(all_labels, all_preds, average="binary")
pixelwise_recall_val = recall_score(all_labels, all_preds)
pixelwise_precision_val = precision_score(all_labels, all_preds)
pixelwise_accuracy_val = accuracy_score(all_labels, all_preds)

# Print validation results
print(f"Validation Loss: {avg_val_loss:.4f} | Mean IOU: {val_metrics['mean_iou']:.4f} | Mean Accuracy: {val_metrics['mean_accuracy']:.4f}")
print(f"Validation F1 Score: {pixelwise_f1_val:.4f}")
print(f"Validation Recall: {pixelwise_recall_val:.4f}")
print(f"Validation Precision: {pixelwise_precision_val:.4f}")
print(f"Validation Accuracy: {pixelwise_accuracy_val:.4f}")

print("Using model on test dataset...")

# Process test images
images = sorted(os.listdir("./data/test"))
images = [im for im in images if im.startswith("test")]
images = [im for im in images if os.path.isdir(f"./data/test/{im}")]

for im in images:
    print(im)
    # Open the image
    image = Image.open(f"./data/test/{im}/{im}.png")

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
    segm_im.save(f"./data/test/{im}/{im}_pred_segformer_ft_geocropdeg_15.png")

print("Test images processed. Creating submission file...")

# Create submission file
submission_filename = 'submission__segformer_ft_geocropdeg_15.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = f"./data/test/test_{i}/test_{i}_pred_segformer_ft_geocropdeg_15.png"
    # print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)

print("Submission file created. Done!")