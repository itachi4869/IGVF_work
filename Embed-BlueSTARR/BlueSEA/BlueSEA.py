import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os
import glob
import time
import gc
import sys
import random
import matplotlib.pyplot as plt

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, counts):
        self.embeddings = embeddings
        self.counts = counts
        
    def __len__(self):
        return len(self.embeddings)
        
    def __getitem__(self, idx):
        x = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self.counts[idx], dtype=torch.float32)
        return x, y

class BlueSEA(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(BlueSEA, self).__init__()
        # Original layers (commented out)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Currently using direct mapping
        self.fc = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # Original forward pass (commented out)
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)
        # Current simplified forward pass
        x = self.fc(x)
        return x

# Add a global set to track which files have been inspected
inspected_files = set()

def load_single_npz_file(file_path):
    """Load embeddings and counts from a single NPZ file"""
    global inspected_files
    
    data = np.load(file_path)
    embeddings = data['embeddings']
    counts = data['counts']
    
    print(f"Loading {os.path.basename(file_path)}...  Embeddings shape: {embeddings.shape}  Counts shape: {counts.shape}")
    
    # Inspect embeddings only if we haven't seen this file before
    if file_path not in inspected_files:
        inspected_files.add(file_path)
        if 'train' in os.path.basename(file_path) or 'validation' in os.path.basename(file_path) or 'test' in os.path.basename(file_path):
            print(f"First time loading this file - inspecting embeddings...")
            inspect_embeddings(embeddings, counts)
    
    return embeddings, counts

def inspect_embeddings(embeddings, counts, num_samples=5):
    """Randomly sample and inspect a few embeddings and their counts"""
    num_samples = min(num_samples, len(embeddings))
    indices = random.sample(range(len(embeddings)), num_samples)
    
    print(f"\n--- Randomly inspecting {num_samples} embeddings ---")
    for i, idx in enumerate(indices):
        embedding = embeddings[idx]
        count = counts[idx]
        
        # Calculate statistics
        zeros_pct = (embedding == 0).sum() / embedding.size * 100
        ones_pct = (embedding == 1).sum() / embedding.size * 100
        between_pct = ((embedding > 0) & (embedding < 1)).sum() / embedding.size * 100
        
        print(f"Sample {i+1}/{num_samples} (idx {idx}):")
        print(f"  Count values: {count}")
        print(f"  Embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}")
        print(f"  Embedding distribution: {zeros_pct:.1f}% zeros, {ones_pct:.1f}% ones, {between_pct:.1f}% between")
        
        # Alert if there are too many 1s or 0s (e.g., > 90%)
        if zeros_pct > 90:
            print(f"  WARNING: This embedding contains {zeros_pct:.1f}% zeros!")
        if ones_pct > 90:
            print(f"  WARNING: This embedding contains {ones_pct:.1f}% ones!")
    
    print("-------------------------------------------\n")

def get_npz_files(embeddings_dir, split='train'):
    """Get list of NPZ files for a specific data split without loading them"""
    split_pattern = f"*{split}*.npz"
    npz_files = sorted(glob.glob(os.path.join(embeddings_dir, split_pattern)))
    
    if not npz_files:
        raise ValueError(f"No NPZ files found for split '{split}' in {embeddings_dir}")
    
    print(f"Found {len(npz_files)} {split} files")
    return npz_files

def calculate_naive_correlation(y_true, y_pred):
    """Calculate metrics using naive correlation method from BERT-STARR"""
    NUM_DNA = 3  # First half of counts are DNA counts

    # Add 1 to counts for smoothing (following BERT-STARR implementation)
    DNA = y_true[:, 0:NUM_DNA] + 1
    RNA = y_true[:, NUM_DNA:] + 1
    
    # Calculate sum of DNA and RNA counts
    sumX = np.sum(DNA, axis=1)
    sumY = np.sum(RNA, axis=1)
    
    # Calculate naive theta (ratio of RNA to DNA)
    naiveTheta = sumY / sumX
    
    # Print statistics
    print("Y true shape:", naiveTheta.shape)
    print("naiveTheta=", naiveTheta[:10])  # Print first 10 values
    print("mean naiveTheta=", np.mean(naiveTheta))
    print("sd naiveTheta=", np.std(naiveTheta))
    print("y_pred=", y_pred[:10])  # Print first 10 values
    print("mean y_pred=", np.mean(y_pred))
    print("sd y_pred=", np.std(y_pred))
    
    # Calculate MSE
    mse = np.square(naiveTheta - y_pred).mean()
    print("MSE=", mse)
    
    # Calculate correlations
    pearson_cor = pearsonr(y_pred.flatten(), naiveTheta.flatten())[0]
    spearman_cor = spearmanr(y_pred.flatten(), naiveTheta.flatten())[0]
    print("Pearson cor=", pearson_cor)
    print("Spearman cor=", spearman_cor)
    
    return {
        'pearson': pearson_cor,
        'spearman': spearman_cor,
        'mse': mse
    }

def train_on_file(model, file_path, batch_size, criterion, optimizer, device):
    """Train the model on data from a single npz file"""
    # Load data from file
    embeddings, counts = load_single_npz_file(file_path)
    
    # Create dataset and dataloader
    dataset = EmbeddingDataset(embeddings, counts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # Similar to BERT-STARR.py, transform count data to target value
        target_ratio = target[:, 0] / target[:, 1]
        
        data, target_ratio = data.to(device), target_ratio.to(device)
        
        # Forward pass
        output = model(data).view(-1)
        loss = criterion(output, target_ratio)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 1000 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - "
                  f"Loss: {loss.item():.6f} - Time: {time.time() - start_time:.2f}s")
            start_time = time.time()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Average loss: {avg_loss:.6f}")
    print()  # Add empty line
    
    # Clean up
    del embeddings
    del counts
    del dataset
    del dataloader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return avg_loss

def evaluate_on_files(model, file_paths, batch_size, criterion, device):
    """Evaluate the model on multiple npz files"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_raw_counts = []  # Store raw counts for naive correlation
    file_count = 0
    
    for file_path in file_paths:
        # Load data from file
        embeddings, counts = load_single_npz_file(file_path)
        
        # Create dataset and dataloader
        dataset = EmbeddingDataset(embeddings, counts)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        # Evaluate
        with torch.no_grad():
            for data, target in dataloader:
                target_ratio = target[:, 0] / target[:, 1]
                data, target_ratio = data.to(device), target_ratio.to(device)
                
                output = model(data).view(-1)
                total_loss += criterion(output, target_ratio).item() * len(target)
                
                all_preds.append(output.cpu().numpy())
                all_raw_counts.append(target.cpu().numpy())  # Store the original counts
        
        file_count += 1
        print(f"Processed test file {file_count}/{len(file_paths)}")
        
        # Clean up
        del embeddings
        del counts
        del dataset
        del dataloader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate metrics
    if all_preds and all_raw_counts:
        all_preds = np.concatenate(all_preds)
        all_raw_counts = np.concatenate(all_raw_counts)
        
        avg_loss = total_loss / len(all_preds)
        print(f"Average Loss: {avg_loss:.6f}")
        
        # Calculate naive correlation metrics
        print("\nCalculating naive correlation metrics:")
        naive_metrics = calculate_naive_correlation(all_raw_counts, all_preds)
        
        return avg_loss, naive_metrics
    else:
        return 0, {}

def main(embeddings_dir=None, batch_size=64, hidden_dim=256, epochs=10, learning_rate=0.001, use_validation=False):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set default embeddings directory if not provided
    if embeddings_dir is None:
        embeddings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embeddings")
    
    # Get list of training files
    train_files = get_npz_files(embeddings_dir, split='train')
    
    # Prepare validation files list if validation is enabled
    validation_files = []
    if use_validation:
        try:
            validation_files = get_npz_files(embeddings_dir, split='validation')
            print(f"Found {len(validation_files)} validation files (will process sequentially during training)")
        except Exception as e:
            print(f"Warning: Could not find validation files: {str(e)}")
            print("Continuing without validation...")
            use_validation = False
    
    # Model initialization will need example data
    # Load a small batch from first file to get dimensions
    sample_embeddings, sample_counts = load_single_npz_file(train_files[0])
    input_dim = sample_embeddings.shape[1]  # Embedding dimension from DeepSEA
    output_dim = 1  # Predicting a single value (ratio of counts)
    del sample_embeddings, sample_counts
    gc.collect()
    
    # Initialize model
    model = BlueSEA(input_dim, hidden_dim, output_dim).to(device)
    print(model)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # To track best validation performance and training losses
    best_val_loss = float('inf')
    best_val_metrics = None
    epoch_losses = []  # Track loss values across epochs
    val_losses = []    # Track validation loss values across epochs
    
    # Create loss log file
    loss_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loss_log.csv")
    with open(loss_log_path, 'w') as loss_file:
        loss_file.write("epoch,train_loss,val_loss\n")  # Write header
    
    # Training
    print(f"Starting training for {epochs} epochs")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train on all files
        epoch_loss = 0
        files_processed = 0
        
        for train_file in train_files:
            print(f"Processing file {os.path.basename(train_file)} ({files_processed+1}/{len(train_files)})")
            file_loss = train_on_file(
                model, train_file, batch_size, criterion, optimizer, device
            )
            epoch_loss += file_loss
            files_processed += 1
        
        # Calculate average epoch loss
        epoch_loss /= len(train_files) if train_files else 1
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.6f}")
        print()  # Add empty line
        
        # Track the loss for plotting
        epoch_losses.append(epoch_loss)
        
        # Validation if enabled - process files one by one
        val_loss = None  # Initialize val_loss for logging
        if use_validation and validation_files:
            print("Evaluating on validation files sequentially...")
            val_loss, val_naive_metrics = evaluate_on_files(
                model, validation_files, batch_size, criterion, device
            )
            
            # Track validation loss for plotting
            val_losses.append(val_loss)
            
            print(f"\nValidation results:")
            print(f"Loss: {val_loss:.6f}")
            print(f"Naive correlation - Pearson: {val_naive_metrics.get('pearson', 'N/A'):.4f}")
            print(f"Naive correlation - Spearman: {val_naive_metrics.get('spearman', 'N/A'):.4f}")
            print(f"Naive correlation - MSE: {val_naive_metrics.get('mse', 'N/A'):.6f}")
        
        # Write losses to CSV file
        with open(loss_log_path, 'a') as loss_file:
            if val_loss is not None:
                loss_file.write(f"{epoch+1},{epoch_loss:.6f},{val_loss:.6f}\n")
            else:
                loss_file.write(f"{epoch+1},{epoch_loss:.6f},\n")  # Empty val_loss field if no validation
        
        # Save best model based on validation loss
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = {
                'loss': val_loss,
                'naive_pearson': val_naive_metrics.get('pearson', None),
                'naive_spearman': val_naive_metrics.get('spearman', None),
                'naive_mse': val_naive_metrics.get('mse', None)
            }
            
            # Save best model
            best_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluesea_model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': best_val_metrics
            }, best_model_path)
            print(f"New best model saved to {best_model_path}!")
        
        print(f"Epoch time: {time.time() - epoch_start_time:.2f}s")
        print("-" * 50)
    
    # Plot training loss and validation loss if available
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, 'o-', color='blue', linewidth=2, markersize=8, label='Training Loss')
    
    # Add validation loss to the plot if validation was used
    if use_validation and val_losses:
        plt.plot(range(1, epochs + 1), val_losses, 's-', color='red', linewidth=2, markersize=8, label='Validation Loss')
    
    plt.title('Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, epochs + 1))
    plt.legend(fontsize=12)
    
    # Add value labels to each point
    for i, loss in enumerate(epoch_losses):
        plt.text(i+1, loss, f'{loss:.6f}', ha='center', va='bottom', fontsize=10)
    
    if use_validation and val_losses:
        for i, loss in enumerate(val_losses):
            plt.text(i+1, loss, f'{loss:.6f}', ha='center', va='top', fontsize=10, color='red')
    
    # Save the plot
    loss_plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_loss.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    
    # Final model save
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluesea_model_final.pth")
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    
    # Print best validation results if validation was used
    if use_validation and best_val_metrics:
        print("\nBest validation performance:")
        print(f"Loss: {best_val_metrics['loss']:.6f}")
        print(f"Naive correlation - Pearson: {best_val_metrics['naive_pearson']:.4f}")
        print(f"Naive correlation - Spearman: {best_val_metrics['naive_spearman']:.4f}")
        if best_val_metrics['naive_mse'] is not None:
            print(f"Naive correlation - MSE: {best_val_metrics['naive_mse']:.6f}")
    
    # Test evaluation
    print("\nEvaluating on test files...")
    test_files = get_npz_files(embeddings_dir, split='test')
    test_loss, test_naive_metrics = evaluate_on_files(
        model, test_files, batch_size, criterion, device
    )
    
    print(f"\nFinal test results (using last model):")
    print(f"Loss: {test_loss:.6f}")
    print(f"Naive correlation - Pearson: {test_naive_metrics.get('pearson', 'N/A'):.4f}")
    print(f"Naive correlation - Spearman: {test_naive_metrics.get('spearman', 'N/A'):.4f}")
    print(f"Naive correlation - MSE: {test_naive_metrics.get('mse', 'N/A'):.6f}")
    
    # If validation was used, also evaluate using the best model
    if use_validation and os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluesea_model_best.pth")):
        print("\nLoading best validation model for test evaluation...")
        checkpoint = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bluesea_model_best.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_naive_metrics = evaluate_on_files(
            model, test_files, batch_size, criterion, device
        )
        
        print(f"\nTest results using best validation model (epoch {checkpoint['epoch']}):")
        print(f"Loss: {test_loss:.6f}")
        print(f"Naive correlation - Pearson: {test_naive_metrics.get('pearson', 'N/A'):.4f}")
        print(f"Naive correlation - Spearman: {test_naive_metrics.get('spearman', 'N/A'):.4f}")
        print(f"Naive correlation - MSE: {test_naive_metrics.get('mse', 'N/A'):.6f}")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="BlueSEA: Train a model on DeepSEA embeddings")
    
    parser.add_argument("--embeddings_dir", type=str, help="Directory with embedding files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--validate", action="store_true", help="Enable validation after each epoch")
    
    args = parser.parse_args()
    
    main(
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        use_validation=args.validate
    )
