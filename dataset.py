import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torchvision.utils

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from typing import Any
import random

class DiverseArtDataSet(Dataset):
    """
    Custom dataset. Indexing into the dataset will return a pair of images and there similarity label.

    Courtesy of https://github.com/maticvl/dataHacker/blob/master/pyTorch/014_siameseNetwork.ipynb
    as this where the image preparation logic comes from. I just modified a small part of it for
    my particular data.
    """
    def __init__(self, image_folder: ImageFolder, transform: transforms.Compose) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.transform = transform
    
    def __getitem__(self, index: int) -> Any:
        image_0_tuple = random.choice(self.image_folder.imgs)

        same_class = random.randint(0, 1)
        if same_class:
            while True:
                image_1_tuple = random.choice(self.image_folder.imgs)
                if image_0_tuple[1] == image_1_tuple[1]:
                    break
        else:
            while True:
                image_1_tuple = random.choice(self.image_folder.imgs)
                if image_0_tuple[1] != image_1_tuple[1]:
                    break
                
        image_0 = Image.open(image_0_tuple[0])
        image_1 = Image.open(image_1_tuple[0])

        image_0 = image_0.convert("RGB")
        image_1 = image_1.convert("RGB")

        image_0_tensor = self.transform(image_0)
        image_1_tensor = self.transform(image_1)

        distance = torch.from_numpy(np.array([int(image_1_tuple[1] != image_0_tuple[1])], dtype=np.float32))

        return image_0_tensor, image_1_tensor, distance
    
    def __len__(self) -> int:
        return len(self.image_folder.imgs)
    
class ArtData:
    """
    Class for handling all of the data. Also includes some functionality for plotting (used for testing).
    """
    def __init__(self, data_path: str) -> None:
        self.image_folder = ImageFolder(root=data_path)

        self.internal_size = (300, 300)
        self.transform = transforms.Compose([
            transforms.Resize(self.internal_size),
            transforms.ToTensor(),
        ])

        self.dataset = DiverseArtDataSet(self.image_folder, self.transform)
        self.data_loader = DataLoader(self.dataset, shuffle=True, batch_size=128, num_workers=6)

    def show_example(self):
        image_0, image_1, distance = self.dataset[0]

        print(distance.item())

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(image_0[0])
        axes[0].axis("off")

        axes[1].imshow(image_1[0])
        axes[1].axis("off")

        plt.show()

    def show_example_batch(self):
        image0_batch, image1_batch, distance_batch = next(iter(self.data_loader))
        images = torch.cat((image0_batch, image1_batch), 0)
        image_grid = torchvision.utils.make_grid(images)
        images_np = image_grid.numpy()

        print(distance_batch.numpy().reshape(-1))

        plt.imshow(np.transpose(images_np, (1, 2, 0)))
        plt.axis("off")
        plt.show()
