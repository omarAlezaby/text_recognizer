import torch
import torch.nn as nn

class LeNet(nn.Module):

    def __init__(self, n_classes):

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            nn.Dropout(.2)
        )

        self.flat = nn.Flatten(dim=1)

        self.classifier = nn.Sequential(
            nn.Linear(12 * 64, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1)
        )

    def __forward__(self, x):
        x = self.features(x)
        x = self.flat(x)
        x = self.classifier(x)
        
        return x
