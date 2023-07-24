import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class FEM_Dataset(Dataset):
    def __init__(self, path, img_size, transform=None):
        self.transform = transform
        df = pd.read_csv(path, header=None)
        self.images = df.iloc[:, 1:7745].values.astype('float32').reshape(-1, img_size, img_size)
        self.labels = df.iloc[:, 0].values.astype('int')
        print('Image size:', self.images.shape)
        print('--- Label ---')
        print(df.iloc[:, 0].value_counts())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = self.images[idx]
        img = Image.fromarray(self.images[idx])

        if self.transform:
            img = self.transform(img)

        return img, label