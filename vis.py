import torch  # PyTorch for deep learning
import glob  # File path handling
import numpy as np  # Numerical computations
import os  # OS operations
import subprocess  # Running shell commands

import torchvision.transforms as transforms  # Image transformations
import torch.nn.functional as F  # Functional API for operations
from torch import nn  # Neural network modules
from models.model import generate_model  # Feature extractor model
from learner import Learner  # Custom classifier
from PIL import Image, ImageFilter, ImageOps, ImageChops  # Image processing utilities
import random  # Randomization utilities
import numbers  # Number utilities
import pdb  # Debugging utilities
import time  # Time measurement
import cv2  # OpenCV for image processing
from matplotlib import pyplot as plt  # Plotting
from tqdm import tqdm  # Progress bar
import sys  # System utilities
import argparse  # Argument parsing

# Try importing accimage for faster image processing
try:
    import accimage
except ImportError:
    accimage = None

# Argument parsing for video anomaly detection
parser = argparse.ArgumentParser(description='Video Anomaly Detection')
parser.add_argument('--n', default='', type=str, help='file name')
args = parser.parse_args()

class ToTensor(object):
    """
    Converts a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts an image in the range [0, 255] to a PyTorch tensor in the range [0.0, 1.0].
    """
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(self.norm_value)
        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = len(pic.mode) if pic.mode not in ['YCbCr', 'I;16'] else 3 if pic.mode == 'YCbCr' else 1
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(self.norm_value)
    
    def randomize_parameters(self):
        pass

class Normalize(object):
    """Normalizes a tensor image with given mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
    
    def randomize_parameters(self):
        pass

#############################################################
#                        MAIN CODE                          #
#############################################################

# Load feature extractor and classifier
model = generate_model()  # Feature extractor
classifier = Learner().cuda()  # Classifier

# Load pre-trained weights
checkpoint = torch.load('./weight/RGB_Kinetics_16f.pth')
model.load_state_dict(checkpoint['state_dict'])
checkpoint = torch.load('./weight/ckpt.pth')
classifier.load_state_dict(checkpoint['net'])

# Set models to evaluation mode
model.eval()
classifier.eval()

# Load image paths
path = args.n + '/*'
save_path = args.n + '_result'
img = glob.glob(path)
img.sort()
segment = len(img) // 16
x_value = [i for i in range(segment)]

# Initialize input tensor
inputs = torch.Tensor(1, 3, 16, 240, 320)
x_time = [jj for jj in range(len(img))]
y_pred = [0] * 16

# Process images and make predictions
for num, i in enumerate(img):
    if num < 16:
        # Fill the input tensor with the first 16 images
        inputs[:, :, num, :, :] = ToTensor(1)(Image.open(i))
    else:
        # Shift the previous frames to the left and add the new frame at index 15
        inputs[:, :, :15, :, :] = inputs[:, :, 1:, :, :]
        inputs[:, :, 15, :, :] = ToTensor(1)(Image.open(i))
        inputs = inputs.cuda()
        
        start = time.time()
        # Forward pass through the feature extractor model
        output, feature = model(inputs)
        feature = F.normalize(feature, p=2, dim=1)  # Normalize feature vectors
        
        # Pass extracted features through the classifier
        out = classifier(feature)
        y_pred.append(out.item())
        end = time.time()
        
        FPS = str(1 / (end - start))[:5]  # Calculate frames per second
        out_str = str(out.item())[:5]  # Extract prediction score
        
        # Read the current frame and annotate it with FPS and prediction score
        cv_img = cv2.imread(i)
        h, w, _ = cv_img.shape
        cv_img = cv2.putText(cv_img, f'FPS: {FPS}, Pred: {out_str}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 240), 2)
        
        # Highlight frames with anomaly predictions above threshold
        if out.item() > 0.4:
            cv_img = cv2.rectangle(cv_img, (0, 0), (w, h), (0, 0, 255), 3)
    
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    # Save the annotated frame
    path = f'./{save_path}/{os.path.basename(i)}'
    cv2.imwrite(path, cv_img)

# Convert frames to video using ffmpeg
os.system(f'ffmpeg -i "{save_path}/%05d.jpg" "{save_path}.mp4"')

# Save prediction plot
plt.plot(x_time, y_pred)
plt.savefig(f'{save_path}.png', dpi=300)
plt.cla()
