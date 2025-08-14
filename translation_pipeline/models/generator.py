import torch
import torch.nn as nn

class Block(nn.Module):
    """
    A basic building block for the U-Net architecture.
    This block consists of a convolutional layer, an optional batch normalization layer,
    and an activation function. It's used for both downsampling (encoder) and
    upsampling (decoder).
    """
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            # The convolutional layer
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # Batch normalization layer
            nn.BatchNorm2d(out_channels),
            # Activation function
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        # Apply dropout if specified
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    The U-Net Generator architecture.
    This model is an encoder-decoder with skip connections between corresponding layers.
    The skip connections help the model pass low-level information (like edges)
    directly across the network, which is crucial for image-to-image translation.
    """
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        ) # Output: 128x128
        self.down1 = Block(features, features * 2, down=True, act="leaky") # 64x64
        self.down2 = Block(features * 2, features * 4, down=True, act="leaky") # 32x32
        self.down3 = Block(features * 4, features * 8, down=True, act="leaky") # 16x16
        self.down4 = Block(features * 8, features * 8, down=True, act="leaky") # 8x8
        self.down5 = Block(features * 8, features * 8, down=True, act="leaky") # 4x4
        self.down6 = Block(features * 8, features * 8, down=True, act="leaky") # 2x2

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1, padding_mode="reflect"), # 1x1
            nn.ReLU(),
        )

        # Decoder
        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features * 8 * 2, features * 8, down=False, act="relu")
        self.up5 = Block(features * 8 * 2, features * 4, down=False, act="relu")
        self.up6 = Block(features * 4 * 2, features * 2, down=False, act="relu")
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu")
        
        # Final output layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1)) # Skip connection from d7
        up3 = self.up3(torch.cat([up2, d6], 1)) # Skip connection from d6
        up4 = self.up4(torch.cat([up3, d5], 1)) # Skip connection from d5
        up5 = self.up5(torch.cat([up4, d4], 1)) # Skip connection from d4
        up6 = self.up6(torch.cat([up5, d3], 1)) # Skip connection from d3
        up7 = self.up7(torch.cat([up6, d2], 1)) # Skip connection from d2

        return self.final_up(torch.cat([up7, d1], 1)) # Skip connection from d1