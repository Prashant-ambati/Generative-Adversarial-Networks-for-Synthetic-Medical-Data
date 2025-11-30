"""
Generator Network for Conditional GAN
Built by Prashant Ambati
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Generator network for creating synthetic medical data.
    
    Args:
        noise_dim (int): Dimension of input noise vector
        condition_dim (int): Dimension of condition vector
        output_dim (int): Dimension of generated data
        hidden_dims (list): List of hidden layer dimensions
    """
    
    def __init__(self, noise_dim=100, condition_dim=10, output_dim=20, hidden_dims=[256, 512, 256]):
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        
        # Input dimension is noise + condition
        input_dim = noise_dim + condition_dim
        
        # Build the network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, noise, condition):
        """
        Forward pass of the generator.
        
        Args:
            noise (torch.Tensor): Random noise vector [batch_size, noise_dim]
            condition (torch.Tensor): Condition vector [batch_size, condition_dim]
            
        Returns:
            torch.Tensor: Generated synthetic data [batch_size, output_dim]
        """
        # Concatenate noise and condition
        x = torch.cat([noise, condition], dim=1)
        
        # Generate synthetic data
        synthetic_data = self.network(x)
        
        return synthetic_data
    
    def generate_samples(self, num_samples, condition, device='cpu'):
        """
        Generate synthetic samples given conditions.
        
        Args:
            num_samples (int): Number of samples to generate
            condition (torch.Tensor): Condition for generation
            device (str): Device to run on
            
        Returns:
            torch.Tensor: Generated synthetic samples
        """
        self.eval()
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, self.noise_dim, device=device)
            
            # Expand condition to match batch size
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).repeat(num_samples, 1)
            
            # Generate samples
            samples = self.forward(noise, condition)
            
        return samples
