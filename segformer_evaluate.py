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
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, jaccard_score
from skimage import morphology
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax
from scipy.ndimage import rotate
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import cv2
from skimage.io import imread
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
          if root.split("/")[-1] not in ["images", "set_rotations", "random_rotations", "flips", "cropresize", "blurs", "noises"]:
            continue
          if root.split("/")[-1] != "images":
            files = [root.split("/")[-1] + "/" + f for f in files]
          image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          if root.split("/")[-1] not in ["groundtruth", "set_rotations", "random_rotations", "flips", "cropresize", "blurs", "noises"]:
            continue
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

images = os.listdir("./data/validation/predictions_segformer")

# Process all validation images into segmentation maps
for im in sorted(images):
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

# Process all validation images
labels = []
preds_standard = []
preds_CRF = []
preds_eroded_dilated = []

diamond = morphology.diamond(3)
diag_coef = 0.8

def edge_power(gradient_x, gradient_y, diagonal=False):
    """ Compute a sense of the power of the edges in the image """
    if diagonal:
        compass_operators = [np.array([[-1, -1,  2], [-1,  2, -1], [ 2, -1, -1]]),
                            np.array([[2, -1,  -1], [-1,  2, -1], [ -1, -1, 2]]),
                            # ... add more compass operators as needed
                            ]
    else:
        compass_operators = [np.array([[-1, -1,  -1], [-1,  8, -1], [ -1, -1, -1]]),
                            # ... add more compass operators as needed
                            ]

    edge_responses = [cv2.filter2D(gradient_x, cv2.CV_64F, kernel) +
                        cv2.filter2D(gradient_y, cv2.CV_64F, kernel)
                        for kernel in compass_operators]
    
    non_max_suppressed = np.max(edge_responses, axis=0)

    threshold = 1
    edges = non_max_suppressed >= threshold

    return sum(edges.flatten())


for im in sorted(images):
    print(f"Evaluating image {im}")
    # Get image, segmentation and ground truth
    image = imread(f"./data/validation/images/{im}")
    segm = imread(f"./data/validation/predictions_segformer/{im}")
    segm_binary = segm > 0
    segm_binary = segm_binary.astype(float)
    segm_rgb = gray2rgb(segm)
    gt = imread(f"./data/validation/groundtruth/{im}")
    gt = np.round(gt/255)

    # Use edge detection to see if the image has a diagonal road
    original_image = cv2.imread(f"./data/validation/predictions_segformer/{im}", cv2.IMREAD_GRAYSCALE)
    gradient_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

    standard_weight = edge_power(gradient_x, gradient_y, diagonal=False)
    diagonal_weight = edge_power(gradient_x, gradient_y, diagonal=True)

    # Get the prediction probabilities
    model.eval() 
    image = Image.open(f"./data/validation/images/{im}")
    # Prepare the image for the model
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=gt.shape, mode="bilinear", align_corners=False)

    p = torch.nn.functional.softmax(upsampled_logits, dim=1)

    p = p[0].cpu().numpy()

    if diagonal_weight >= diag_coef*standard_weight:
        # Add zero padding around
        p_pad = np.pad(p, ((0,0),(400,400),(400,400)), mode='constant')
        segm_pad = np.pad(segm, ((400,400),(400,400)), mode='constant')

        # Use the CRF to get the final prediction
        # Get the unary from the softmax output
        W, H, NLABELS = segm_pad.shape[0], segm_pad.shape[1], 2
        U = unary_from_softmax(p_pad)
    else:
        W, H, NLABELS = segm.shape[0], segm.shape[1], 2
        U = unary_from_softmax(p)

    # Horizontal
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(35, 1), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(50):
        d.stepInference(Q, tmp1, tmp2)
    kl1 = d.klDivergence(Q) / (H*W)
    map_soln_h = np.argmax(Q, axis=0).reshape((H,W))

    # Vertical
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(1, 35), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(50):
        d.stepInference(Q, tmp1, tmp2)
    kl2 = d.klDivergence(Q) / (H*W)
    map_soln_v = np.argmax(Q, axis=0).reshape((H,W))

    map_soln = map_soln_h + map_soln_v


    if diagonal_weight >= diag_coef*standard_weight:
        print(f"Image {im} has a diagonal edge")
        # Evaluate the CRF with different rotations
        rotations = [45]
        for rot in rotations:
            # Rotate the probs by rot degrees
            rotated_p = np.array([rotate(p_pad[i], rot, reshape=False) for i in range(2)])
            rotated_U = unary_from_softmax(rotated_p)

            # Horizontal
            d = dcrf.DenseCRF2D(W, H, NLABELS)
            d.setUnaryEnergy(rotated_U)
            d.addPairwiseGaussian(sxy=(40, 1), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q, tmp1, tmp2 = d.startInference()
            for _ in range(50):
                d.stepInference(Q, tmp1, tmp2)
            kl1 = d.klDivergence(Q) / (H*W)
            map_soln_1 = np.argmax(Q, axis=0).reshape((H,W))
            map_soln_1 = rotate(map_soln_1, -rot, reshape=False)

            # # Vertical
            d = dcrf.DenseCRF2D(W, H, NLABELS)
            d.setUnaryEnergy(rotated_U)
            d.addPairwiseGaussian(sxy=(1, 40), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q, tmp1, tmp2 = d.startInference()
            for _ in range(50):
                d.stepInference(Q, tmp1, tmp2)
            kl2 = d.klDivergence(Q) / (H*W)
            map_soln_2 = np.argmax(Q, axis=0).reshape((H,W))
            map_soln_2 = rotate(map_soln_2, -rot, reshape=False)

            map_soln = map_soln + map_soln_1 + map_soln_2

    # Together
    map_soln = map_soln > 0
    soln_array = np.array(map_soln)
    soln_array = soln_array.astype(float)
    if diagonal_weight >= diag_coef*standard_weight:
        soln_array = soln_array[400:-400, 400:-400]

    eroded = morphology.binary_erosion(soln_array, diamond)
    eroded_dilated = morphology.binary_dilation(eroded, diamond)

    # Get the metrics
    labels.extend(gt.flatten())
    preds_standard.extend(segm_binary.flatten())
    preds_CRF.extend(soln_array.flatten())
    preds_eroded_dilated.extend(eroded_dilated.flatten())

def evaluate_segmentation(pred, gt):
    """ Evaluate segmentation results """
    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, pos_label=1)
    recall = recall_score(gt, pred, pos_label=1)
    precision = precision_score(gt, pred, pos_label=1)
    mean_iou = jaccard_score(gt, pred, average='macro')
    print(f"Accuracy: {accuracy}")
    print(f"F1: {f1}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Mean IoU: {mean_iou}")

evaluate_segmentation(preds_eroded_dilated, labels)

print("Using model on test dataset...")

# Process test images
images_test = sorted(os.listdir("./data/test"))
images_test = [im for im in images_test if im.startswith("test")]
images_test = [im for im in images_test if os.path.isdir(f"./data/test/{im}")]


# Post process test set

images_test = sorted(os.listdir("./data/test"))
images_test = [im for im in images_test if im.startswith("test")]
images_test = [im for im in images_test if os.path.isdir(f"./data/test/{im}")]

diamond = morphology.diamond(3)
diag_coef = 0.8

for im in images_test:
    print(f"Evaluating image {im}")
    # Open the image and the segmentation map
    image = imread(f"./data/test/{im}/{im}.png")
    segm = imread(f"./data/test/{im}/{im}_pred_segformer_ft_geocropdeg_15.png")
    segm_binary = segm > 0
    segm_binary = segm_binary.astype(float)
    segm_rgb = gray2rgb(segm)

    # Use edge detection to see if the image has a diagonal road
    original_image = cv2.imread(f"./data/test/{im}/{im}_pred_segformer_ft_geocropdeg_15.png", cv2.IMREAD_GRAYSCALE)
    gradient_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)

    standard_weight = edge_power(gradient_x, gradient_y, diagonal=False)
    diagonal_weight = edge_power(gradient_x, gradient_y, diagonal=True)

    # Get the prediction probabilities
    model.eval() 
    image = Image.open(f"./data/test/{im}/{im}.png")
    # Prepare the image for the model
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=segm.shape, mode="bilinear", align_corners=False)

    p = torch.nn.functional.softmax(upsampled_logits, dim=1)
    p = p[0].cpu().numpy()

    # Add zero padding around
    p_pad = np.pad(p, ((0,0),(400,400),(400,400)), mode='constant')
    segm_pad = np.pad(segm, ((400,400),(400,400)), mode='constant')

    if diagonal_weight >= diag_coef*standard_weight:
        # Add zero padding around
        p_pad = np.pad(p, ((0,0),(400,400),(400,400)), mode='constant')
        segm_pad = np.pad(segm, ((400,400),(400,400)), mode='constant')

        # Use the CRF to get the final prediction
        # Get the unary from the softmax output
        W, H, NLABELS = segm_pad.shape[0], segm_pad.shape[1], 2
        U = unary_from_softmax(p_pad)
    else:
        W, H, NLABELS = segm.shape[0], segm.shape[1], 2
        U = unary_from_softmax(p)

    # Horizontal
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(35, 1), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(50):
        d.stepInference(Q, tmp1, tmp2)
    kl1 = d.klDivergence(Q) / (H*W)
    map_soln_h = np.argmax(Q, axis=0).reshape((H,W))

    # Vertical
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(1, 35), compat=15, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(50):
        d.stepInference(Q, tmp1, tmp2)
    kl2 = d.klDivergence(Q) / (H*W)
    map_soln_v = np.argmax(Q, axis=0).reshape((H,W))

    map_soln = map_soln_h + map_soln_v

    if diagonal_weight >= diag_coef*standard_weight:
        print(f"Image {im} has a diagonal edge")
        # Evaluate the CRF with different rotations
        rotations = [45]
        for rot in rotations:
            # Rotate the probs by rot degrees
            rotated_p = np.array([rotate(p_pad[i], rot, reshape=False) for i in range(2)])
            rotated_U = unary_from_softmax(rotated_p)

            # Horizontal
            d = dcrf.DenseCRF2D(W, H, NLABELS)
            d.setUnaryEnergy(rotated_U)
            d.addPairwiseGaussian(sxy=(40, 1), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q, tmp1, tmp2 = d.startInference()
            for _ in range(50):
                d.stepInference(Q, tmp1, tmp2)
            kl1 = d.klDivergence(Q) / (H*W)
            map_soln_1 = np.argmax(Q, axis=0).reshape((H,W))
            map_soln_1 = rotate(map_soln_1, -rot, reshape=False)

            # # Vertical
            d = dcrf.DenseCRF2D(W, H, NLABELS)
            d.setUnaryEnergy(rotated_U)
            d.addPairwiseGaussian(sxy=(1, 40), compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
            Q, tmp1, tmp2 = d.startInference()
            for _ in range(50):
                d.stepInference(Q, tmp1, tmp2)
            kl2 = d.klDivergence(Q) / (H*W)
            map_soln_2 = np.argmax(Q, axis=0).reshape((H,W))
            map_soln_2 = rotate(map_soln_2, -rot, reshape=False)

            map_soln = map_soln + map_soln_1 + map_soln_2

    # Together
    map_soln = map_soln > 0
    soln_array = np.array(map_soln)
    soln_array = soln_array.astype(float)
    if diagonal_weight >= diag_coef*standard_weight:
        soln_array = soln_array[400:-400, 400:-400]

    # Eroding - dilating for final improvements
    eroded = morphology.binary_erosion(soln_array, diamond)
    soln_array = morphology.binary_dilation(eroded, diamond)

    soln_array = soln_array*255
    soln_array = soln_array.astype(np.uint8)

    # Save the segmentation map
    segm_im = Image.fromarray(soln_array)
    segm_im.save(f"./data/test/{im}/{im}_postprocessed2_pred_segformer_ft_geocropdeg_15.png")

print("Test images processed. Creating submission file...")

# Create submission file
submission_filename = 'submission_postprocessed2_segformer_ft_geocropdeg_15.csv'
image_filenames = []
for i in range(1, 51):
    image_filename = f"./data/test/test_{i}/test_{i}_postprocessed2_pred_segformer_ft_geocropdeg_15.png"
    # print(image_filename)
    image_filenames.append(image_filename)
masks_to_submission(submission_filename, *image_filenames)

print("Submission file created. Done!")