import torch
import torch.nn.functional as F
from federated_learning.config.config import *

def test(model, test_dataset):
    print("\n=== Testing Model ===")
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}, Error Rate: {1 - accuracy:.4f}")
    return 1 - accuracy

def client_update(client_model, optimizer, train_loader, epochs):
    print(f"\n=== Client Training ({epochs} epochs) ===")
    client_model.train()
    criterion = torch.nn.NLLLoss()
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    return client_model.state_dict()

def train_vae(vae, gradients):
    print("\n=== Training VAE ===")
    print(f"Training VAE on {len(gradients)} gradients")
    gradient_dataset = torch.stack(gradients)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    vae.train()
    num_epochs = 100  # Increase epochs for better convergence with larger dataset
    batch_size = min(BATCH_SIZE, len(gradients) // 10)  # Adjust batch size based on dataset size
    dataset = torch.utils.data.TensorDataset(gradient_dataset)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        batches = 0
        
        for (batch,) in data_loader:
            batch = batch.to(device)
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss, recon_loss, kl_loss = vae_loss_with_components(recon_batch, batch, mu, logvar)
            loss.backward()
            vae_optimizer.step()
            
            epoch_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            batches += 1
        
        # Print more detailed metrics
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"VAE Epoch {epoch + 1}/{num_epochs}, " +
                  f"Total Loss: {epoch_loss/batches:.4f}, " +
                  f"Recon Loss: {recon_loss_sum/batches:.4f}, " +
                  f"KL Loss: {kl_loss_sum/batches:.4f}")

def vae_loss_with_components(recon_x, x, mu, logvar):
    """Returns the VAE loss along with its components for monitoring"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kld
    return total_loss, recon_loss, kld

def train_dual_attention(dual_attention, feature_vectors, labels):
    print("\n=== Training Dual Attention ===")
    print(f"Training Dual Attention on {len(feature_vectors)} feature vectors")
    print(f"Class distribution: {sum(labels.numpy() == 0)} benign, {sum(labels.numpy() == 1)} malicious")
    
    # Weight loss based on class distribution to handle imbalance
    pos_weight = torch.tensor([(sum(labels.numpy() == 0) / sum(labels.numpy() == 1))]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Positive weight for loss: {pos_weight.item():.4f}")
    
    attention_optimizer = torch.optim.Adam(dual_attention.parameters(), lr=0.001)
    dual_attention.train()
    num_epochs = 150  # Increase epochs for better learning on the complex dataset
    batch_size = min(BATCH_SIZE, len(feature_vectors) // 20)  # Smaller batches for better generalization
    dataset = torch.utils.data.TensorDataset(feature_vectors, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Training for {num_epochs} epochs with batch size {batch_size}")
    
    best_loss = float('inf')
    best_epoch = 0
    early_stop_count = 0
    patience = 20  # Early stopping patience
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            
            attention_optimizer.zero_grad()
            trust_scores = dual_attention(X_batch, X_batch.mean(dim=0, keepdim=True))
            loss = criterion(trust_scores, y_batch)
            loss.backward()
            attention_optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predicted = torch.sigmoid(trust_scores) >= 0.5
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = epoch_loss / len(data_loader)
        accuracy = correct / total
        
        # Print metrics periodically
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Dual Attention Epoch {epoch + 1}/{num_epochs}, " +
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with loss {best_loss:.4f}")
            break
    
    print(f"Dual Attention training completed. Final accuracy: {accuracy:.4f}") 