from torch import nn


class AlexNet(nn.Module):
    """Fake LeNet with 32x32 color images and 200 classes"""

    def __init__(self, num_classes: int = 200) -> None:
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
       
        self.lin1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.lin2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.lin3 = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.lin1(out)
        out = self.lin2(out)
        out = self.lin3(out)
        return out
