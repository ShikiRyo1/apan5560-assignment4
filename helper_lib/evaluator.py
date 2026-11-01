import torch


def evaluate_model(model, data_loader, criterion, device="cpu"):
    model.to(device)
    was_training = model.training
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            n = labels.size(0)
            loss_sum += criterion(logits, labels).item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += n
    model.train(was_training)
    if count == 0:
        raise ValueError("Evaluation loader is empty")
    return loss_sum / count, correct / count
