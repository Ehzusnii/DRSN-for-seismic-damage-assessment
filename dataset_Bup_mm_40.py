# mm
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import numpy as np
import torch



class Dataset_mm(Dataset):
    def __init__(self, images_path: list, images_class: list,outsize:int ,transform=None):
        self.images = images_path
        self.label = images_class
        self.transform = transform
        self.outsize=outsize

    def __getitem__(self, index):
        npz_name_path = self.images[index]

        npz = np.load(npz_name_path)
        npz_arr = npz['arr']
        final_size = 28
        R = npz_arr[0]/10
        G = npz_arr[1]/10
        B = npz_arr[2]/10
        
        R = cv2.resize(R, dsize=None, fx=final_size / self.outsize, fy=final_size / self.outsize, interpolation=cv2.INTER_AREA)
        G = cv2.resize(G, dsize=None, fx=final_size / self.outsize, fy=final_size / self.outsize, interpolation=cv2.INTER_AREA)
        B = cv2.resize(B, dsize=None, fx=final_size / self.outsize, fy=final_size / self.outsize, interpolation=cv2.INTER_AREA)
        
      
        RGB = np.array([R, G, B])
        npz_tensor = torch.from_numpy(RGB)
        imgg = npz_tensor.to(torch.float32)

        if self.transform is not None:
            imgg = self.transform(imgg)


        label = self.label[index]
        label = torch.from_numpy(np.array(label)).long()

        return imgg, label

    def __len__(self):
        return len(self.label)