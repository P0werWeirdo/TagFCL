import numpy as np
from PIL import Image
import cv2
import os
import pandas as pd

class DistillDataset:
    def __init__(self, train_data, train_label):
        self.TrainData = train_data
        self.TrainLabels = train_label
        self.OutputLogits = []

    def add_output_logits(self, output_logits):
        self.OutputLogits = output_logits

    def __getitem__(self, index):
        if self.OutputLogits != []:
            return index, self.TrainData[index], self.TrainLabels[index], self.OutputLogits[index]
        else:
            return index, self.TrainData[index], self.TrainLabels[index]

    def __len__(self):
        return len(self.TrainData)