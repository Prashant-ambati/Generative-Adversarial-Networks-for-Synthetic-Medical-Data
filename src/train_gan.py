"""
Training Script for Conditional GAN on Medical Data
Built by Prashant Ambati
"""

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
try:
    import joblib
except Exception:
    joblib = None
import pickle
from tqdm import tqdm

from models.conditional_gan import ConditionalWGAN
from data.data_loader import MedicalDataLoader
from evaluation.statistical_tests import StatisticalEvaluator


def train_gan(args):
    """
    Train the Conditional GAN on medical data.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = MedicalDataLoader(random_state=args.seed)
    train_loader, test_loader, original_df = data_loader.create_data_loaders(
        batch_size=args.batch_size,
        data_path=args.data_path
    )
    
    # Get data dimensions
    sample_batch = next(iter(train_loader))
    data_dim = sample_batch[0].shape[1]
    condition_dim = sample_batch[1].shape[1]
    
    print(f"Data dimension: {data_dim}")
    print(f"Condition dimension: {condition_dim}")
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Initialize GAN
    gan = ConditionalWGAN(
        noise_dim=args.noise_dim,
        condition_dim=condition_dim,
        data_dim=data_dim,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_gp=args.lambda_gp,
        device=device
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_wd = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (real_data, conditions) in enumerate(progress_bar):
            real_data = real_data.to(device)
            conditions = conditions.to(device)
            
            # Train GAN
            losses = gan.train_step(real_data, conditions, n_critic=args.n_critic)
            
            epoch_g_loss += losses['g_loss']
            epoch_d_loss += losses['d_loss']
            epoch_wd += losses['wasserstein_distance']
            
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f"{losses['g_loss']:.4f}",
                'D_Loss': f"{losses['d_loss']:.4f}",
                'WD': f"{losses['wasserstein_distance']:.4f}"
            })
        
        # Calculate epoch averages
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        avg_wd = epoch_wd / len(train_loader)
        
        print(f"Epoch {epoch+1}: G_Loss={avg_g_loss:.4f}, D_Loss={avg_d_loss:.4f}, WD={avg_wd:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            gan.save_model(f'checkpoints/gan_epoch_{epoch+1}.pth')
            print(f"Model saved at epoch {epoch+1}")
        
        # Generate and evaluate synthetic data
        if (epoch + 1) % args.eval_interval == 0:
            evaluate_synthetic_data(gan, test_loader, data_loader, device, epoch+1)
    
    # Final model save
    os.makedirs('models', exist_ok=True)
    gan.save_model('models/final_gan_model.pth')
    print("Training completed! Final model saved.")

    # Save preprocessing artifacts
    artifacts = {
        'scaler': data_loader.scaler,
        'label_encoders': data_loader.label_encoders,
        'onehot_encoder': getattr(data_loader, 'onehot_encoder', None)
    }
    preprocess_path = 'models/preprocessing.joblib'
    try:
        if joblib is not None:
            joblib.dump(artifacts, preprocess_path)
        else:
            with open(preprocess_path, 'wb') as f:
                pickle.dump(artifacts, f)
        print(f"Preprocessing artifacts saved to {preprocess_path}")
    except Exception as e:
        print(f"Failed to save preprocessing artifacts: {e}")
    
    # Final evaluation
    print("Performing final evaluation...")
    evaluate_synthetic_data(gan, test_loader, data_loader, device, args.epochs, final=True)
    
    # Plot training history
    plot_training_history(gan.history)


def evaluate_synthetic_data(gan, test_loader, data_loader, device, epoch, final=False):
    """
    Evaluate the quality of synthetic data.
    
    Args:
        gan: Trained GAN model
        test_loader: Test data loader
        data_loader: Data loader instance
        device: Computing device
        epoch: Current epoch
        final: Whether this is final evaluation
    """
    # Get test data
    real_data_list = []
    conditions_list = []
    
    for real_data, conditions in test_loader:
        real_data_list.append(real_data.numpy())
        conditions_list.append(conditions.numpy())
    
    real_data = np.vstack(real_data_list)
    conditions = np.vstack(conditions_list)
    
    # Generate synthetic data
    conditions_tensor = torch.FloatTensor(conditions).to(device)
    synthetic_data = gan.generate_synthetic_data(len(real_data), conditions_tensor)
    synthetic_data = synthetic_data.cpu().numpy()
    
    # Statistical evaluation
    evaluator = StatisticalEvaluator()
    
    # Feature names (simplified for demonstration)
    feature_names = [
        'age', 'bmi', 'bp_systolic', 'bp_diastolic', 'cholesterol',
        'glucose', 'heart_rate', 'temperature', 'resp_rate', 'oxygen_sat',
        'wbc', 'rbc', 'hemoglobin', 'platelets', 'creatinine',
        'sodium', 'potassium', 'chloride', 'co2', 'bun'
    ]
    
    results = evaluator.comprehensive_evaluation(real_data, synthetic_data, feature_names)
    
    print(f"Epoch {epoch} - Quality Score: {results['quality_score']:.4f}")
    
    if final:
        # Save detailed evaluation report
        os.makedirs('results', exist_ok=True)
        report = evaluator.generate_report('results/evaluation_report.txt')
        print("Detailed evaluation report saved to results/evaluation_report.txt")
        
        # Save synthetic data sample
        synthetic_df = data_loader.inverse_transform_data(synthetic_data)
        np.savetxt('results/synthetic_data_sample.csv', synthetic_df[:1000], delimiter=',')
        print("Synthetic data sample saved to results/synthetic_data_sample.csv")


def plot_training_history(history):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Generator loss
    axes[0].plot(history['g_loss'])
    axes[0].set_title('Generator Loss')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss')
    
    # Discriminator loss
    axes[1].plot(history['d_loss'])
    axes[1].set_title('Discriminator Loss')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Loss')
    
    # Wasserstein distance
    axes[2].plot(history['wasserstein_distance'])
    axes[2].set_title('Wasserstein Distance')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Distance')
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training history plot saved to results/training_history.png")


def main():
    parser = argparse.ArgumentParser(description='Train Conditional GAN for Medical Data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_g', type=float, default=0.0001, help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=0.0004, help='Discriminator learning rate')
    parser.add_argument('--noise_dim', type=int, default=100, help='Noise dimension')
    parser.add_argument('--lambda_gp', type=float, default=10, help='Gradient penalty coefficient')
    parser.add_argument('--n_critic', type=int, default=5, help='Critic updates per generator update')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None, help='Path to medical data CSV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Logging parameters
    parser.add_argument('--save_interval', type=int, default=25, help='Model save interval')
    parser.add_argument('--eval_interval', type=int, default=20, help='Evaluation interval')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Train the GAN
    train_gan(args)


if __name__ == '__main__':
    main()
