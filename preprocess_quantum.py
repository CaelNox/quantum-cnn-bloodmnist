import os
import torch
import numpy as np
from tqdm import tqdm
from utils.dataset import get_dataloaders
from models.quanv_cnn import QuantumFilter

def extract_and_cache_quantum_features():
    """
    Extracts 4-channel quantum feature maps (14x14x4) for BloodMNIST images
    using the 4-qubit QuantumFilter and caches tensors to outputs/.
    """
    os.makedirs("outputs", exist_ok=True)
    train_loader, val_loader, test_loader, _, _ = get_dataloaders(batch_size=1)
    q_filter = QuantumFilter()

    def process_split(loader, split_name):
        feature_list = []
        label_list = []

        print(f"\nProcessing Quantum Features for {split_name} split...")
        for img, lbl in tqdm(loader):
            # Ensure shape is 2D (28, 28)
            img_np = img.squeeze().numpy()
            if img_np.ndim == 3:
                img_np = img_np.mean(axis=0)  # Average channels if RGB

            # Normalize to [0, 1] for RY angle encoding
            img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            # Output grid: 14x14 patches, 4 features per patch -> shape (4, 14, 14)
            q_map = np.zeros((4, 14, 14))

            for row in range(0, 28, 2):
                for col in range(0, 28, 2):
                    patch = img_norm[row:row+2, col:col+2].flatten()
                    exp_vals = q_filter.process_patch(patch)
                    for channel in range(4):
                        q_map[channel, row//2, col//2] = exp_vals[channel]

            feature_list.append(torch.tensor(q_map, dtype=torch.float32))
            label_list.append(lbl.squeeze())

        features_tensor = torch.stack(feature_list)
        labels_tensor = torch.stack(label_list)
        return features_tensor, labels_tensor

    q_train_x, q_train_y = process_split(train_loader, "TRAIN")
    q_val_x, q_val_y = process_split(val_loader, "VAL")
    q_test_x, q_test_y = process_split(test_loader, "TEST")

    cache_data = {
        'train_x': q_train_x, 'train_y': q_train_y,
        'val_x': q_val_x, 'val_y': q_val_y,
        'test_x': q_test_x, 'test_y': q_test_y
    }

    torch.save(cache_data, "outputs/quantum_features.pt")
    print("\n[SUCCESS] Quantum Features Cached to outputs/quantum_features.pt!")

if __name__ == "__main__":
    extract_and_cache_quantum_features()