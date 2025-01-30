#torch
import torch
import torch.nn as nn
import torch.nn.functional as F


# LSTM each sequence and CNN per slice independently -> flatten and combined -> fcl to prediction
class LSTM_CNN_2D(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, channels, kernel_size, num_filters, dropout):
        super(LSTM_CNN_2D, self).__init__()
                
        # LSTM will process 64 sequences (one per pixel location) of 24 values (depth)
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # CNN will process each depth slice independently
        self.conv = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout))
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 16 * lstm_hidden_size + (16//2) * (16//2) * num_filters * 24, 1024),  
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256))
        
    def forward(self, x):
        self.batch_size, self.depth, self.height, self.width = x.size()
        line_profiles = x.permute(0,2,3,1)
        
        # Reshape for LSTM
        # Convert input to (batch_size * 64, depth, 1) - 64 sequences of 24 depth values
        lstm_input = line_profiles.reshape(self.batch_size * self.height * self.width, self.depth, 1)
        # LSTM feature extraction for each pixel location across the depth
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :]  # Take the last hidden state
        lstm_features_flat = lstm_features.view(self.batch_size, self.height, self.width, -1)  # Reshape to (batch_size, height, width, lstm_hidden_size)
        
        # CNN feature extraction
        cnn_features_list = []
        for slice_idx in range(self.depth):
            # Take a slice and add channel dimension
            cnn_input = x[:, slice_idx, :, :].unsqueeze(1)
            # Apply CNN and flatten the output
            cnn_output = self.conv(cnn_input).view(self.batch_size, -1)
            cnn_features_list.append(cnn_output)
        
         # Combine features from all depth slices
        cnn_features_combined = torch.cat(cnn_features_list, dim=1)  # Shape: (batch_size, num_filters * height * width * depth)
        cnn_features_combined = cnn_features_combined.view(self.batch_size, self.height, self.width, -1)  # Reshape to (batch_size, height, width, num_filters * depth)
        
        # Combine LSTM and CNN features
        combined_features = torch.cat((lstm_features_flat, cnn_features_combined), dim=-1)  # Combine along the feature dimension
        combined_features = combined_features.view(self.batch_size, -1)  # Flatten spatial dimensions
        
        # Final prediction
        final_predictions = self.fc(combined_features).view(self.batch_size, self.height, self.width)
        
        return final_predictions, line_profiles
           
    def get_name(self):
        return 'LSTM_CNN_2D'
    
# LSTM each sequence and CNN across all stack -> flatten and combined in the bottleneck -> fcl to prediction
class LSTM_CNN_3D(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTM_CNN_3D, self).__init__()
        
        # LSTM will process sequences of 3D patches
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # 3D CNN Encoder based on Generator's encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU())
        self.encoder2 = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 4),
            nn.ReLU())
        self.encoder3 = nn.Sequential(
            nn.Conv3d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 8),
            nn.ReLU())
        
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(hidden_dim * 8, hidden_dim * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 16),
            nn.ReLU())

        # 3D CNN Decoder based on Generator's decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d((hidden_dim * 2 + 16 * 16) * lstm_hidden_size, hidden_dim * 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 16),
            nn.ReLU())
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d((hidden_dim * 16) + 8 * 16, hidden_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 8),
            nn.ReLU())
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d((hidden_dim * 8 ) + 8 * 8, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(hidden_dim * 2),
            nn.ReLU())
        self.decoder4 = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim * 3, input_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(input_dim),
            nn.Tanh())
        
        # Fully connected layers for the final output
        self.fc = nn.Sequential(
            nn.Linear(output_dim * output_dim * 24, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256))
         
        self.output_dim = output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, depth, height, width = x.size()
        line_profiles = x.permute(0,2,3,1)
        
        # Process each depth slice with LSTM
        lstm_input = line_profiles.reshape(batch_size * height * width, depth, 1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_features = lstm_out[:, -1, :].view(batch_size, -1, 1, 1, 1)  # Shape: (batch_size, lstm_hidden_size, 1, 1, 1)
        
        # Encode the 3D image patch using 3D CNN Encoder
        x = x.unsqueeze(1)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        # Bottleneck layer to ensure the output is (batch_size, lstm_hidden_size, 1, 1, 1)
        bottleneck = self.bottleneck_conv(enc3)

        # Combine image features and LSTM features
        combined_features = torch.cat((bottleneck, lstm_features), dim=1)

        ###fix decoder parte
        # Decode with skip connections using 3D CNN Decoder
        dec1 = self.decoder1(combined_features)
        dec1_ups = F.interpolate(dec1, size=(3, 2, 2), mode='trilinear', align_corners=False)
        dec1_cat = torch.cat((dec1_ups, enc3), dim=1)  # Skip connection
        dec2 = self.decoder2(dec1_cat)
        dec2 = torch.cat((dec2, enc2), dim=1)  # Skip connection
        dec3 = self.decoder3(dec2)
        dec3 = torch.cat((dec3, enc1), dim=1)  # Skip connection
        dec4 = self.decoder4(dec3)
        
        # Collapse the depth dimension and pass through fully connected layers
        dec4 = dec4.squeeze(1).view(batch_size, -1)
        output_flat = self.fc(dec4)
        output = output_flat.view(dec4.size(0), self.output_dim, self.output_dim)
        
        return output, lstm_features
    
    def get_name(self):
        return 'LSTM_CNN_3D'
    
# model = LSTM_CNN_3D(lstm_input_size=1, lstm_hidden_size=8, lstm_num_layers=1, input_dim=1, hidden_dim=16, output_dim=16)
# x = torch.randn(128, 24, 16, 16)  # Example input # Example input
# pred = model(x)

 ###debugging
# model = LSTM_CNN_FLAT(lstm_input_size=1, lstm_hidden_size=12, lstm_num_layers=1, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
# random_tensor = torch.rand(192,24,16,16)
# model(random_tensor)
