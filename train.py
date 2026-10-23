import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from utils.dataset import get_dataloaders
from models.quanv_cnn import QuanvClassifier
from models.classical_cnn import ClassicalCNN
from utils.metrics import compute_metrics, plot_comparison_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            # Ensure target is 1D tensor of long integers
            y_batch = y_batch.squeeze(-1).to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(DEVICE)
                y_v = y_v.squeeze(-1).to(DEVICE).long()
                preds = model(x_v).argmax(dim=1)
                val_correct += (preds == y_v).sum().item()
                val_total += y_v.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")

    return model

def evaluate_model(model, test_loader, num_classes=8):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x_t, y_t in test_loader:
            x_t = x_t.to(DEVICE)
            y_t = y_t.squeeze(-1).long()
            
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_t.numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs), num_classes)
    return metrics, np.array(all_preds), np.array(all_labels)

def main():
    print(f"Using Device: {DEVICE}")
    os.makedirs("outputs", exist_ok=True)

    # 1. Load Raw BloodMNIST for Classical Baseline
    c_train_loader, c_val_loader, c_test_loader, num_classes, class_labels = get_dataloaders(batch_size=64)

    # 2. Train Classical Baseline
    print("\n" + "="*50)
    print("TRAINING CLASSICAL CNN BASELINE")
    print("="*50)
    classical_model = ClassicalCNN(num_classes=num_classes)
    classical_model = train_model(classical_model, c_train_loader, c_val_loader, epochs=10)
    c_metrics, c_preds, y_true = evaluate_model(classical_model, c_test_loader, num_classes)

    # 3. Load Cached Quantum Features
    if not os.path.exists("outputs/quantum_features.pt"):
        print("\n[ERROR] Quantum features not found. Run 'python preprocess_quantum.py' first!")
        return

    print("\nLoading Cached Quantum Features...")
    q_data = torch.load("outputs/quantum_features.pt")
    
    q_train_ds = TensorDataset(q_data['train_x'], q_data['train_y'])
    q_val_ds = TensorDataset(q_data['val_x'], q_data['val_y'])
    q_test_ds = TensorDataset(q_data['test_x'], q_data['test_y'])

    q_train_loader = DataLoader(q_train_ds, batch_size=64, shuffle=True)
    q_val_loader = DataLoader(q_val_ds, batch_size=64, shuffle=False)
    q_test_loader = DataLoader(q_test_ds, batch_size=64, shuffle=False)

    # 4. Train Quantum CNN Head
    print("\n" + "="*50)
    print("TRAINING HYBRID QUANTUM CNN (QUANVOLUTIONAL)")
    print("="*50)
    quanv_model = QuanvClassifier(num_classes=num_classes)
    quanv_model = train_model(quanv_model, q_train_loader, q_val_loader, epochs=10)
    q_metrics, q_preds, _ = evaluate_model(quanv_model, q_test_loader, num_classes)

    # 5. Print Metrics Summary Table
    print("\n" + "="*60)
    print("FINAL TEST METRICS COMPARISON (BLOODMNIST 8-CLASS)")
    print("="*60)
    print(f"{'Metric':<20} | {'Classical CNN':<15} | {'Quantum CNN (Quanv)':<15}")
    print("-" * 60)
    for k in ['accuracy', 'macro_f1', 'weighted_f1', 'roc_auc']:
        print(f"{k.upper():<20} | {c_metrics[k]:<15.4f} | {q_metrics[k]:<15.4f}")
    print("="*60)

    # 6. Plot Confusion Matrices
    plot_comparison_matrix(y_true, q_preds, c_preds, class_labels)

if __name__ == "__main__":
    main()