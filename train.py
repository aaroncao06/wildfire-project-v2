import torch
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, roc_auc_score

# Define the training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_num = 0
    print('start')
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
        batch_num+=1
        print(f'batch #{batch_num} done')

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy

def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = torch.round(outputs).squeeze()

            # Collect the true and predicted labels
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predicted.cpu().tolist())

    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    # Calculate evaluation metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)

    return f1, precision, accuracy, recall, auc_roc