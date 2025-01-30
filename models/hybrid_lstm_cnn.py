import torch
import torch.nn as nn

class HYBRID_LSTM_CNN_FLAT(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, channels, kernel_size, num_filters, dropout):
        super(HYBRID_LSTM_CNN_FLAT, self).__init__()

        # LSTM configuration
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # CNN configuration
        self.conv = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding= kernel_size//2, stride=1),
            nn.ReLU(),
            #nn.MaxPool2d(1),
            nn.AvgPool2d(2),
            nn.Dropout(dropout),
        )

        # Fully connected layers for final output reduction
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * 24 + (16//2) * (16//2) * num_filters * 24, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)  # Output for each pixel location
        )
        
    def forward(self, x):
        batch_size, depth, height, width = x.size()
        line_profiles = x.permute(0,2,3,1)
        
        # LSTM feature extraction
        lstm_input = x.view(batch_size, depth, height * width)  # Prepare input for LSTM
        lstm_out, _ = self.lstm(lstm_input)  # lstm_out shape: [batch_size, height * width, depth]

        # CNN feature extraction
        cnn_features_list = []
        for slice_idx in range(depth):
            cnn_input = x[:, slice_idx, :, :].unsqueeze(1)
            cnn_output = self.conv(cnn_input)
            
            # Expand LSTM output for each slice to match CNN spatial dimensions
            lstm_output_flat = lstm_out[:, slice_idx, :].view(batch_size, -1)   #[batch_size, lstm_hidden_size, h, w]
            
            # Flatten and combine CNN output with expanded LSTM slice output
            cnn_output_flat = cnn_output.view(batch_size, -1)
            combined_features_slice = torch.cat((cnn_output_flat, lstm_output_flat), dim=1)
            cnn_features_list.append(combined_features_slice)

        # Concatenate all combined features from all slices
        combined_features = torch.cat(cnn_features_list, dim=1)

        # Final prediction
        final_predictions = self.fc(combined_features).view(batch_size, height, width)

        return final_predictions, line_profiles
    
    def get_name(self):
        return 'HYBRID_LSTM_CNN_FLAT'
    
class HYBRID_LSTM_CNN_SLICE(nn.Module):
    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, channels, kernel_size, num_filters, dropout):
        super(HYBRID_LSTM_CNN_SLICE, self).__init__()

        # LSTM configuration
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        
        # CNN configuration
        self.conv = nn.Sequential(
            nn.Conv2d(channels, num_filters, kernel_size=kernel_size, padding=kernel_size//2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.AvgPool2d(2),
            nn.Dropout(dropout),
        )

        # Convolutional layers for combined feature maps
        self.combined_conv = nn.Sequential(
            nn.Conv2d(num_filters + lstm_hidden_size, 8, kernel_size=3, padding=kernel_size//2, stride=1),  # Combining both features
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fully connected layer for final output reduction
        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 24, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256))

    def forward(self, x):
        batch_size, depth, height, width = x.size()
        line_profiles = x.permute(0,2,3,1)
        
        # LSTM feature extraction
        lstm_input = x.view(batch_size, depth, height * width) # Prepare input for LSTM
        lstm_out, _ = self.lstm(lstm_input)  # lstm_out shape: [batch_size, height * width, lstm_hidden_size]
    
        # CNN feature extraction
        combined_features_list = []
        for slice_idx in range(depth):
            cnn_input = x[:, slice_idx, :, :].unsqueeze(1)
            cnn_output = self.conv(cnn_input)
            
            # Expand LSTM output for each slice to match CNN spatial dimensions
            lstm_slice_output = lstm_out[:, slice_idx, :].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
            
            # Combine CNN output with expanded LSTM slice output
            cat_features = torch.cat((cnn_output, lstm_slice_output), dim=1)
            combined_features = self.combined_conv(cat_features)
            
            # Flatten for final layer processing
            combined_features_flat = combined_features.view(batch_size, -1)
            combined_features_list.append(combined_features_flat)

        # Concatenate all combined and processed features from all slices
        final_combined_features = torch.cat(combined_features_list, dim=1)

        # Final prediction
        final_predictions = self.fc(final_combined_features).view(batch_size, height, width)

        return final_predictions, line_profiles
    
    def get_name(self):
        return 'HYBRID_LSTM_CNN_SLICE'
    
#debugging
# model = HYBRID_LSTM_CNN_SLICE(lstm_input_size=256, lstm_hidden_size=256, lstm_num_layers=2, channels=1, kernel_size=3, num_filters=8, dropout=0.1)
# random_tensor = torch.rand(64,24,16,16)
# model(random_tensor)