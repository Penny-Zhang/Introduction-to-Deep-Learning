"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl

""" Parts of the U-Net model """
"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# ###第二个模型
# class DoubleConv(nn.Module):

#     def __init__(self, in_channels,mid_channels, out_channels):
#         super(DoubleConv, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1, bias=False),
#             # Same conv. not valid conv. in original paper
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.n_channels = 3
        self.n_classes = num_classes
        self.bilinear = hparams["bilinear"]

        self.inc = DoubleConv(3, 30)
        self.down1 = Down(30, 60)
        self.down2 = Down(60, 120)
        self.down3 = Down(120, 240)
        self.down4 = Down(240, 240)
        self.up1 = Up(480, 120, self.bilinear)
        self.up2 = Up(240, 60, self.bilinear)
        self.up3 = Up(120, 30, self.bilinear)
        self.up4 = Up(60, 30, self.bilinear)
        self.outc = OutConv(30, num_classes)
        
        # ####第二个模型
        # self.original_channels = 3
        # self.num_classes = num_classes

        # self.encoder1 = DoubleConv(self.original_channels,mid_channels=30, out_channels=30)
        # self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.encoder2 = DoubleConv(in_channels=30, mid_channels=60,out_channels=60)
        # self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.encoder3 = DoubleConv(in_channels=60, mid_channels=120,out_channels=120)
        # self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.encoder4 = DoubleConv(in_channels=120, mid_channels=240,out_channels=240)
        # # self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # # self.encoder5 = DoubleConv(in_channels=512,mid_channels=1024, out_channels=1024)

        # # self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # # self.decoder1 = DoubleConv(in_channels=1024, mid_channels=512,out_channels=512)

        # self.up2 = nn.ConvTranspose2d(in_channels=240, out_channels=120, kernel_size=2, stride=2)
        # self.decoder2 = DoubleConv(in_channels=240,mid_channels=120, out_channels=120)

        # self.up3 = nn.ConvTranspose2d(in_channels=120, out_channels=60, kernel_size=2, stride=2)
        # self.decoder3 = DoubleConv(in_channels=120,mid_channels=60, out_channels=60)

        # self.up4 = nn.ConvTranspose2d(in_channels=60, out_channels=30, kernel_size=2, stride=2)
        # self.decoder4 = DoubleConv(in_channels=60, mid_channels=30,out_channels=30)


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################


    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        # ####第二个模型
        # encoder1 = self.encoder1(x)             
        # encoder1_pool = self.down1(encoder1)     
        # encoder2 = self.encoder2(encoder1_pool) 
        # encoder2_pool = self.down2(encoder2)   
        # encoder3 = self.encoder3(encoder2_pool) 
        # encoder3_pool = self.down3(encoder3)    
        # encoder4 = self.encoder4(encoder3_pool)  
        # # encoder4_pool = self.down4(encoder4)    
        # # encoder5 = self.encoder5(encoder4_pool)  

        # # decoder1_up = self.up1(encoder5)        
        # # decoder1 = self.decoder1(torch.cat((encoder4, decoder1_up), dim=1))
        # #                                          

        # decoder2_up = self.up2(encoder4)         
        # decoder2 = self.decoder2(torch.cat((encoder3, decoder2_up), dim=1))                  

        # decoder3_up = self.up3(decoder2)        
        # decoder3 = self.decoder3(torch.cat((encoder2, decoder3_up), dim=1))
                                               
        # decoder4_up = self.up4(decoder3)         
        # decoder4 = self.decoder4(torch.cat((encoder1, decoder4_up), dim=1))
                                                 
        # x = self.decoder5(decoder4)            

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
