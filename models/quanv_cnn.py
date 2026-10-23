import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

class QuantumFilter:
    """
    4-Qubit Quanvolutional Filter.
    Takes a 2x2 patch (4 normalized pixels) and applies:
    1. RY(pi * x_i) Angle Encoding
    2. Circular CNOT Entanglement
    3. RZ(pi * x_i) Phase Shift
    4. Expectation values <Z> across all 4 qubits.
    """
    def __init__(self):
        # Pauli Z observables for 4 qubits
        self.observables = [
            SparsePauliOp("IIIZ"),
            SparsePauliOp("IIZI"),
            SparsePauliOp("IZII"),
            SparsePauliOp("ZIII")
        ]

    def process_patch(self, patch):
        """
        patch: numpy array of 4 pixel values in range [0, 1]
        returns: list of 4 expectation values in range [-1, 1]
        """
        qc = QuantumCircuit(4)

        # 1. Angle Encoding
        for i in range(4):
            qc.ry(np.pi * patch[i], i)

        # 2. Entanglement Layer
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)

        # 3. Non-linear Phase Rotation
        for i in range(4):
            qc.rz(np.pi * patch[i], i)

        # Statevector computation (exact simulation)
        state = Statevector.from_instruction(qc)
        exp_vals = [state.expectation_value(obs).real for obs in self.observables]
        return exp_vals

class QuanvClassifier(nn.Module):
    """
    Downstream PyTorch Classification Head operating on cached (14x14x4) Quantum Feature Maps.
    """
    def __init__(self, num_classes=8):
        super(QuanvClassifier, self).__init__()
        # Input shape from quantum filter: (batch_size, 4, 14, 14)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # (16, 7, 7)
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # (32, 4, 4)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits