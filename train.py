import torch

# Define the training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_features, batch_target in train_loader:
        batch_features = batch_features.to(device)
        batch_target = batch_target.to(device)

        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total_loss += loss.item()
        total += batch_target.size(0)
        correct += (predicted == batch_target).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy
