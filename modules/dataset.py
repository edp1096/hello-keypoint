import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class FacialKeypointsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transforms=None):
        self.df = dataframe
        self.transforms = transforms

        self.images = []
        self.keypoints = []

        for index, row in self.df.iterrows():
            # Image
            image = row["Image"]
            image = np.fromstring(image, sep=' ').reshape([96, 96])
            image = np.stack((image, image, image), axis=-1)
            image = image / 255.0

            # Keypoints
            keypoints = row.drop(["Image"])
            keypoints = keypoints.to_numpy().astype("float32")

            # Add to Dataset's images and keypoints
            self.images.append(image)
            self.keypoints.append(keypoints)

    def __getitem__(self, idx):
        #         item: pd.Series = self.df.iloc[idx]

        #         # Image
        #         image = item['Image']
        #         image = np.fromstring(image, sep=' ').reshape([96, 96])
        #         image = np.stack((image, image, image), axis=-1)
        #         image = image / 255.0

        #         # Keypoints
        #         keypoints = item.drop(['Image'])
        #         keypoints = keypoints.to_numpy().astype('float32')

        image = self.images[idx]
        keypoints = self.keypoints[idx]

        return_dict = {"image": image, "keypoints": keypoints}

        if self.transforms:
            return_dict = self.transforms(return_dict)

        return return_dict

    def __len__(self):
        return len(self.df)
