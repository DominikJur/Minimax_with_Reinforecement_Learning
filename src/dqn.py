import torch
import torch.nn as nn




class DQNNetwork(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=1):
        super(DQNNetwork, self).__init__()
        
        # Input processing
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # Two hidden layers with one residual connection
        self.hidden1 = nn.Linear(hidden_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Output pathway
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Input processing
        x = torch.relu(self.input_norm(self.input_layer(x)))
        identity = x
        
        # Hidden layers with one residual connection
        x = torch.relu(self.norm1(self.hidden1(x)))
        x = self.dropout(x)
        x = torch.relu(self.norm2(self.hidden2(x)))
        
        # One residual connection (helps with training)
        x = x + identity
        
        # Output
        return self.output_layer(x)
