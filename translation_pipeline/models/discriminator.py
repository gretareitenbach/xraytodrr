import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    """
    A convolutional block for the Discriminator.
    This block consists of a convolutional layer, an optional batch normalization layer,
    and a LeakyReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=False,
                padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    The PatchGAN Discriminator model.
    It takes two images (input and target/generated) concatenated together and
    determines if each patch of the image is real or fake.

    Args:
        in_channels (int): The number of channels in the input images (e.g., 1 for grayscale).
        features (list): A list of feature map sizes for each convolutional layer.
    """
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            stride = 1 if feature == features[-1] else 2
            layers.append(
                CNNBlock(in_channels, feature, stride=stride)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        The forward pass for the discriminator.

        Args:
            x (Tensor): The input image (e.g., the X-ray).
            y (Tensor): The image to be judged (either the real DRR or the generated DRR).

        Returns:
            Tensor: A feature map where each value is a prediction for a patch of the input.
        """
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

