"""
Test suite for GAN models
Built by Prashant Ambati
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.generator import Generator
from models.discriminator import Discriminator
from models.conditional_gan import ConditionalWGAN


class TestGenerator:
    """Test cases for Generator model."""
    
    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = Generator(noise_dim=100, condition_dim=10, output_dim=20)
        assert generator.noise_dim == 100
        assert generator.condition_dim == 10
        assert generator.output_dim == 20
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        generator = Generator(noise_dim=100, condition_dim=10, output_dim=20)
        
        batch_size = 32
        noise = torch.randn(batch_size, 100)
        condition = torch.randn(batch_size, 10)
        
        output = generator(noise, condition)
        
        assert output.shape == (batch_size, 20)
    
    def test_generator_sample_generation(self):
        """Test sample generation method."""
        generator = Generator(noise_dim=100, condition_dim=10, output_dim=20)
        
        condition = torch.randn(10)
        samples = generator.generate_samples(50, condition)
        
        assert samples.shape == (50, 20)


class TestDiscriminator:
    """Test cases for Discriminator model."""
    
    def test_discriminator_initialization(self):
        """Test discriminator initialization."""
        discriminator = Discriminator(input_dim=20, condition_dim=10)
        assert discriminator.input_dim == 20
        assert discriminator.condition_dim == 10
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        discriminator = Discriminator(input_dim=20, condition_dim=10)
        
        batch_size = 32
        data = torch.randn(batch_size, 20)
        condition = torch.randn(batch_size, 10)
        
        output = discriminator(data, condition)
        
        assert output.shape == (batch_size, 1)
    
    def test_gradient_penalty(self):
        """Test gradient penalty computation."""
        discriminator = Discriminator(input_dim=20, condition_dim=10)
        
        batch_size = 16
        real_data = torch.randn(batch_size, 20)
        fake_data = torch.randn(batch_size, 20)
        condition = torch.randn(batch_size, 10)
        
        gp = discriminator.compute_gradient_penalty(real_data, fake_data, condition)
        
        assert isinstance(gp, torch.Tensor)
        assert gp.dim() == 0  # Scalar


class TestConditionalWGAN:
    """Test cases for Conditional WGAN."""
    
    def test_wgan_initialization(self):
        """Test WGAN initialization."""
        wgan = ConditionalWGAN(
            noise_dim=100,
            condition_dim=10,
            data_dim=20,
            device='cpu'
        )
        
        assert wgan.generator is not None
        assert wgan.discriminator is not None
        assert wgan.optimizer_g is not None
        assert wgan.optimizer_d is not None
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        wgan = ConditionalWGAN(
            noise_dim=100,
            condition_dim=10,
            data_dim=20,
            device='cpu'
        )
        
        num_samples = 100
        conditions = torch.randn(num_samples, 10)
        
        synthetic_data = wgan.generate_synthetic_data(num_samples, conditions)
        
        assert synthetic_data.shape == (num_samples, 20)
    
    def test_training_step(self):
        """Test single training step."""
        wgan = ConditionalWGAN(
            noise_dim=100,
            condition_dim=10,
            data_dim=20,
            device='cpu'
        )
        
        batch_size = 32
        real_data = torch.randn(batch_size, 20)
        conditions = torch.randn(batch_size, 10)
        
        losses = wgan.train_step(real_data, conditions, n_critic=1)
        
        assert 'g_loss' in losses
        assert 'd_loss' in losses
        assert 'wasserstein_distance' in losses
        assert isinstance(losses['g_loss'], float)
        assert isinstance(losses['d_loss'], float)


if __name__ == "__main__":
    pytest.main([__file__])
