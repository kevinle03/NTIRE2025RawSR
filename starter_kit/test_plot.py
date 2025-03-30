import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from degradations import q_sample
from imutils import demosaic, plot_all

raw = np.load('/home/lil61/Documents/SwinIR/combined_diffusion/data/val_in/lr/6.npz')
raw_img = raw["raw"]
raw_max = raw["max_val"]
raw_img = (raw_img / raw_max).astype(np.float32)    
values = [float(line.strip()) for line in open('/home/lil61/Documents/SwinIR/noise.txt') if line.strip()]
values = torch.tensor(values)
t = torch.randint(0, 11, (1,))
img = q_sample(raw_img, torch.tensor(t, dtype=int), values)
raw_img = demosaic(raw_img, 'RGGB')
plot_all ([raw_img], axis='off')