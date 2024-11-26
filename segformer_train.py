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

# Set training parameters
batch_size = 4
epochs = 10

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

# Load dataset
root_dir = './data'
image_processor = SegformerImageProcessor(reduce_labels=False, do_resize=False)

train_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor)
valid_dataset = SemanticSegmentationDataset(root_dir=root_dir, image_processor=image_processor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

# Create dataloaders
train_dataloader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
                              generator=torch.Generator().manual_seed(seed))
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                         num_labels=2,
)

# Load evaluation metric
metric = evaluate.load("mean_iou")

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize W&B run
wandb.login()
job_type = "train_model"
config = {
    "optimizer": "adam",
    "batch_size": batch_size,
    "epochs": epochs,
}
run = wandb.init(project="ml_p2", job_type=job_type, config=config)

# Init best f1 score so far
best_f1 = 0.0

# Begin training
print("Starting training")
model.train()
for epoch in range(epochs):  # loop over the dataset multiple times
  print("Epoch:", epoch+1)
  model.train()
  for idx, batch in enumerate(tqdm(train_dataloader)):
    # Get the inputs;
    pixel_values = batch["pixel_values"].to(device)
    labels = batch["labels"].div(255).round().long().to(device)
    # labels = torch.where(batch["labels"] == 255, 0, 1).to(device)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss, logits = outputs.loss, outputs.logits

    loss.backward()
    optimizer.step()

    # Evaluate
    with torch.no_grad():
      upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

    # Print loss and metrics every 100 batches
    if idx % 100 == 0:
      metrics = metric._compute(
              predictions=predicted.cpu(),
              references=labels.cpu(),
              num_labels=2,
              ignore_index=None,
              reduce_labels=False,
          )

      
      print("Loss:", loss.item())
      # print("Accuracy:", metrics["accuracy"])
      # print("F1 score:", metrics["f1"])
      # print("Precision:", metrics["precision"])
      # print("Recall:", metrics["recall"])
      print("Mean_iou:", metrics["mean_iou"])
      print("Pixel-wise accuracy:", metrics["overall_accuracy"])
      wandb.log({"Epoch": epoch, "Train accuracy": metrics["overall_accuracy"], "Train loss": loss.item(), "Mean IoU": metrics["mean_iou"]})
      #wandb.log({"Epoch": epoch, "Train accuracy": metrics['accuracy'], "Train loss": loss.item(), "Train F1": metrics['f1'], "Train recall": metrics['recall'], "Train precision": metrics['precision']})
  
  all_preds = []
  all_labels = []

  model.eval()  # Set model to evaluation mode
  validation_loss = 0.0
  with torch.no_grad():
    for val_batch in tqdm(valid_dataloader, desc="Validation"):
      # Get the inputs
      pixel_values = val_batch["pixel_values"].to(device)
      # labels = torch.where(val_batch["labels"] == 255, 0, 1).to(device)
      labels = val_batch["labels"].div(255).round().long().to(device)

      # Forward pass
      outputs = model(pixel_values=pixel_values, labels=labels)
      val_loss, logits = outputs.loss, outputs.logits
      validation_loss += val_loss.item()

      # Evaluate validation metrics
      upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
      predicted = upsampled_logits.argmax(dim=1)

      # clf_metrics.add_batch(predictions=predicted.detach().cpu().numpy().flatten(), references=labels.detach().cpu().numpy().flatten())
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


  # Compute pixel-wise scores for the entire validation set
  all_preds = np.concatenate(all_preds)
  all_labels = np.concatenate(all_labels)
  pixelwise_f1_val = f1_score(all_labels, all_preds, average="binary")
  pixelwise_recall_val = recall_score(all_labels, all_preds)
  pixelwise_precision_val = precision_score(all_labels, all_preds)
  pixelwise_accuracy_val = accuracy_score(all_labels, all_preds)
  # print(f"Validation Loss: {avg_val_loss:.4f}")
  print(f"Validation Loss: {avg_val_loss:.4f} | Mean IOU: {val_metrics['mean_iou']:.4f} | Mean Accuracy: {val_metrics['mean_accuracy']:.4f}")
  print(f"Validation F1 Score: {pixelwise_f1_val:.4f}")
  print(f"Validation Recall: {pixelwise_recall_val:.4f}")
  print(f"Validation Precision: {pixelwise_precision_val:.4f}")
  print(f"Validation Accuracy: {pixelwise_accuracy_val:.4f}")

  wandb.log({"Epoch": epoch+1, "Val accuracy": pixelwise_accuracy_val, "Val loss": avg_val_loss, "Val F1": pixelwise_f1_val, "Val recall": pixelwise_recall_val, "Val precision": pixelwise_precision_val, "Mean accuracy": val_metrics['mean_accuracy'], "Mean IoU": val_metrics['mean_iou']})

  if pixelwise_f1_val > best_f1:
    best_f1 = pixelwise_f1_val
    print("Saving best model")
    torch.save(model.state_dict(), f"./models/finetuned_segformer_{epoch+1}.pth") # save model
    torch.save(optimizer.state_dict(), f"./models/finetuned_segformer_optimizer_{epoch+1}.pth") # save optimizer
    wandb.save(f"./models/finetuned_segformer_{epoch+1}.pth")

wandb.finish()