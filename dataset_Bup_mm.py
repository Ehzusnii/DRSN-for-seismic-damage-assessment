# mm
# import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import torch



class Dataset_mm(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images = images_path
        self.label = images_class
        self.transform = transform

    def __getitem__(self, index):
        npz_name_path = self.images[index]

        npz = np.load(npz_name_path)
        npz_arr = npz['arr']
        # final_size = 100
        R = npz_arr[0]/10
        G = npz_arr[1]/10
        B = npz_arr[2]/10
        RGB = np.array([R, G, B])
        npz_tensor = torch.from_numpy(RGB)
        img = npz_tensor.to(torch.float32)

        if self.transform is not None:
            img = self.transform(img)


        label = self.label[index]
        label = torch.from_numpy(np.array(label)).long()

        return img, label

    def __len__(self):
        return len(self.label)