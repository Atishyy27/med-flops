import torch
import torch.nn as nn
import torch.optim as optim

def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    patience=5,
    min_delta=0.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    writer=None,
    global_step=0
):
    """
    Train `model` on train_loader for `epochs`, optionally validating on val_loader.
    Returns the trained model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"[Train] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", avg_loss, global_step + epoch)
            
        if val_loader is not None:
            val_loss, _ = evaluate_model(model, val_loader, device=device, tag=f" Val after epoch {epoch+1}", writer=writer, global_step=global_step + epoch)
            if val_loss + min_delta < best_val:
                best_val = val_loss
                no_improve = 0
            else:
                no_improve += 1

            if writer is not None:
                writer.add_scalar("EarlyStopping/no_improve", no_improve, global_step + epoch)

            if no_improve >= patience:
                print(f"[EarlyStopping] No improvement in {patience} epochs. Stopping early at epoch {epoch+1}.")
                break

    return model, global_step + epochs

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    tag: str = "Test",
    writer=None,
    global_step=0
):
    """
    Evaluate `model` on data_loader and print accuracy.
    Returns (loss, accuracy).
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(data_loader)
    acc = correct / total * 100
    print(f"[{tag}] Loss: {avg_loss:.4f} — Acc: {acc:.2f}%")
    
    if writer is not None:
        writer.add_scalar(f"Loss/{tag}", avg_loss, global_step)
        writer.add_scalar(f"Acc/{tag}", acc, global_step)
        
    return avg_loss, acc
