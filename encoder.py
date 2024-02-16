import torch
import torch.nn as nn
import torch.optim as optim

from dataset import ArtData
from loss import ContrastiveLoss

class SiameseEncoder(nn.Module):
    """
    Siamese network for image embedding.
    """
    def __init__(self) -> None:
        super(SiameseEncoder, self).__init__()

        self.cnn = nn.Sequential(
            # (3, 300, 300)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            # (512, 12, 12)
            nn.Linear(512 * 12 * 12, 2048),
            nn.ReLU(inplace=True),

            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 32)
        )

    def forward_once(self, x: torch.Tensor):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) # Flatten
        output = self.fc(output)
        return output
    
    def forward(self, input_1, input_2):
        output1 = self.forward_once(input_1)
        output2 = self.forward_once(input_2)

        return output1, output2
    
class SiameseTrainer:
    """
    Trainer class for training with the Siamese network.
    """
    def __init__(
            self,
            data_path:  str,
            save_path:  str,
            save_every: int,
            num_epochs: int,
            lr:         float
            ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_path = data_path
        self.save_path = save_path
        self.save_every = save_every
        self.num_epochs = num_epochs
        self.lr = lr

        self.data = ArtData(self.data_path)
        self.num_batches = len(self.data.data_loader)

        self.model = SiameseEncoder().to(self.device)
        self.criterion = ContrastiveLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            self._per_epoch(epoch)

            if (epoch + 1) % self.save_every == 0:
                self._save_checkpoint(epoch + 1)

        self._save_checkpoint(self.num_epochs)
        print(f"Training Finished. Saved parameters to {self.save_path}")

    def _per_epoch(self, epoch: int) -> None:
        for i, (image_0, image_1, label) in enumerate(self.data.data_loader):
            image_0, image_1, label = image_0.to(self.device), image_1.to(self.device), label.to(self.device)

            self.optimizer.zero_grad()

            output_1, output_2 = self.model(image_0, image_1)

            loss = self.criterion(output_1, output_2, label)
            loss.backward()

            self.optimizer.step()

            if i % 25 == 0:
                print(f"epoch: {epoch+1}/{self.num_epochs}, step: {i+1}/{self.num_batches}, loss: {loss.item():.4f}")
    
    def _save_checkpoint(self, epoch: int) -> None:
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Checkpoint saved at epoch {epoch}")