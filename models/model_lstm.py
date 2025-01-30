#torch
import torch
import torch.nn as nn

#model (64 sequences each)
class LSTM_64(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_64, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 24, 1024),
                                nn.ReLU(),
                                nn.Linear(1024,512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, output_dim))
    
    def forward(self, x):
        batch_size, depth, height, width = x.size()
        line_profiles = x.permute(0,2,3,1)
        
        lstm_input = x.view(batch_size, depth, height * width)
        lstm_out, _ = self.lstm(lstm_input)
        lin_out = self.fc(lstm_out.contiguous().view(batch_size, -1))  
        pred = lin_out.view(batch_size, height, width)
        return pred, line_profiles
    
    def get_name(self):
        return 'LSTM_64'
    
# LSTM (independent sequences)
class LSTM_1(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_1, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 24 * 64, 8192),
                                nn.ReLU(),
                                nn.Linear(8192, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 256),
                                nn.ReLU(),
                                nn.Linear(256, output_dim))
    
    def forward(self, x):
        batch_size, depth, height, width = x.size()
        line_profiles = x.permute(0,2,3,1)

        lstm_input = x.view(batch_size * height * width, depth, 1)
        lstm_out, _= self.lstm(lstm_input)
        lstm_features = lstm_out.contiguous().view(batch_size, height * width, -1)
        lin_out = self.fc(lstm_features.view(batch_size, -1))  # Use the last hidden state (last time step) as the output
        pred = lin_out.view(batch_size, height, width)
        return pred, line_profiles
    
    def get_name(self):
        return 'LSTM_1'