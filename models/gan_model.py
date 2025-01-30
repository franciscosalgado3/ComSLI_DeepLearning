import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_CNN_FLAT(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, channels, kernel_size, num_filters, dropout):
        super(LSTM_CNN_FLAT, self).__init__()
                
        # LSTM will process 64 sequences (one per pixel location) of 24 values (depth)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # CNN will process each depth slice independently
        self.conv = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout))
        


#encoder-decoder (u-net)
class Generator(nn.Module):
    def __init__(self, hidden_dim, input_channels, output_dim, dropout):
        super(Generator, self).__init__()
        
        # Encoder for 3D image patch + hint (concatenated as an additional channel)
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.encoder2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.encoder3 = nn.Sequential(
            nn.Conv3d(hidden_dim*2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout))

        # Depth reduction via global average pooling over depth
        self.depth_pooling = nn.AdaptiveAvgPool3d((1, None, None))
        
        # Decoder with 2D transposed convolutions (deconvolutions)
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1))
            # No activation function here (linear activation)
            
    def forward(self, img, angles, hint):        
        # work on input vectors dimensions
        img = img.unsqueeze(1)   # Shape: [batch_size, 1, depth, height, width]
        angles = angles.unsqueeze(1).unsqueeze(2).expand(-1, -1, img.size(2), -1, -1)  # Shape: [batch_size, 1, 24, height, width]
        hint = hint.unsqueeze(1).unsqueeze(2).expand(-1, -1, img.size(2), -1, -1)    # Shape: [batch_size, 1, 24, height, width]
                
        # Concatenate img, angles_3d, and hint_3d along the channel dimension
        x = torch.cat([img, angles, hint], dim=1)  # Shape: [batch_size, 3, depth, height, width]
        
        # 3D Convolutions in the encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        
        # Depth pooling to collapse depth dimension
        x = self.depth_pooling(enc3)  # Shape: [batch_size, hidden_dim*4, 1, height//4, width//4]
        x = x.squeeze(2)  # Remove the depth dimension: [batch_size, hidden_dim*4, height//4, width//4]
       
        # 2D Transposed Convolutions in the decoder
        dec1 = self.decoder1(x)  # Output shape: [batch_size, hidden_dim*2, height//2, width//2]
        dec2 = self.decoder2(dec1)  # Output shape: [batch_size, hidden_dim, height, width]
        dec3 = self.decoder3(dec2)  # Output shape: [batch_size, 1, height, width]
        
        # Remove the channel dimension to get output shape: [batch_size, height, width]
        output = dec3.squeeze(1)
        return output # Final output shape: [batch_size, height, width]
    
    def get_name(self):
        return 'GAN_model'
    
    
class Discriminator(nn.Module):
    def __init__(self,  input_channels, hidden_dim, output_dim, dropout):
        super(Discriminator, self).__init__()
        
        # Convolutional layers for the combined input (generated/real values + hint)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, output_dim, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, gen_output, hint, mask):
        gen_output = gen_output.unsqueeze(1)  # Shape: [batch_size, 1, height, width]
        hint = hint.unsqueeze(1)    # Shape: [batch_size, 1, height, width]
        mask = mask.unsqueeze(1)    # Shape: [batch_size, 1, height, width]

        
        # Concatenate img, gen_output_3d, and hint_3d along the channel dimension
        x = torch.cat([gen_output, hint, mask], dim=1)  # Shape: [batch_size, 2, height, width]
       
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x3 = x3 + x2  # Residual connection
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x5 = x5.squeeze(1)
        return x5

# dubugging
# generator = Generator(input_channels=3, hidden_dim=16, output_dim=1, dropout=0.2)
# x_source = torch.randn(8, 24, 16, 16)  # Example input
# x_pred = torch.randn(8, 16, 16)  # Example LSTM output
# x_hint_vector = torch.randn(8, 16, 16)  # Example hint vector
# pred = generator(x_source, x_pred, x_hint_vector) 

# discriminator = Discriminator(input_channels=2, hidden_dim=16, output_dim=1, dropout=0.1)
# target = torch.randn(8, 16, 16)  # Example input
# hint_vector = torch.randn(8, 16, 16)  # Example hint vector
# mask = torch.randn(8, 16, 16)  # Example mask
# pred = discriminator(target, hint_vector, mask) 
  