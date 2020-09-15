from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch

class dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        csv_dir = os.path.join(root_dir, csv_file)
        csv = pd.read_csv(csv_dir)
        self.labels = csv.iloc[:,0]
        self.images = csv.iloc[:,1:]
        self.images = (self.images.values).reshape(-1,28,28).astype('float32')
        self.labels = torch.tensor(self.labels)
        if transform:
            for i in range(self.images.shape[0]):
                self.images[i] = transform(self.images[i])
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        
        img = self.images[index]
        label = self.labels[index]
    
        sample = {'image': img, 'label': label}
        
        return sample
