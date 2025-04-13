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

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

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
    gradient_dataset = torch.stack(gradients)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    vae.train()
    num_epochs = 50
    batch_size = BATCH_SIZE
    for epoch in range(num_epochs):
        epoch_loss = 0
        permutation = torch.randperm(gradient_dataset.size(0))
        for i in range(0, gradient_dataset.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch = gradient_dataset[indices].to(device)
            vae_optimizer.zero_grad()
            recon_batch, mu, logvar = vae(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            vae_optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"VAE Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/len(gradient_dataset):.4f}")

def train_dual_attention(dual_attention, feature_vectors, labels):
    print("\n=== Training Dual Attention ===")
    attention_optimizer = torch.optim.Adam(dual_attention.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    dual_attention.train()
    num_epochs = 50
    batch_size = BATCH_SIZE
    dataset = torch.utils.data.TensorDataset(feature_vectors, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float()
            attention_optimizer.zero_grad()
            trust_scores = dual_attention(X_batch, X_batch.mean(dim=0, keepdim=True))
            loss = criterion(trust_scores, y_batch)
            loss.backward()
            attention_optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Dual Attention Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/len(data_loader):.4f}") 