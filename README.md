# Hybrid Quantum Convolutional Neural Network (QCNN) on BloodMNIST

A hybrid Quantum-Classical Convolutional Neural Network (Quanvolutional Network) implemented in **Qiskit** and **PyTorch** for 8-class peripheral blood cell morphology classification using the **BloodMNIST** dataset.

---

## Overview & Architecture

Standard 2D convolution filters apply linear weighted sums across pixel patches. In contrast, this Quanvolutional architecture embeds local $2 \times 2$ pixel patches into a multi-qubit quantum state space to extract non-linear spatial correlations:

1. **Quantum Feature Map (4-Qubit Circuit):**
   - $2 \times 2$ patches from $28 \times 28$ grayscale images are mapped to 4 qubits via $RY(\pi x)$ angle encoding.
   - Circular $CX$ (CNOT) entangling gates entangle single-qubit states in a 16-dimensional Hilbert space ($\mathbb{C}^{16}$).
   - Pauli-Z expectation values ($\langle Z \rangle$) are computed across all 4 qubits, generating $14 \times 14 \times 4$ quantum feature maps.

2. **Classification Head:**
   - A PyTorch CNN processes the extracted quantum feature maps to predict logits across 8 blood cell classes (Basophils, Eosinophils, Erythroblasts, Immature Granulocytes, Lymphocytes, Monocytes, Neutrophils, Platelets).

---

## Benchmark Results (BloodMNIST 8-Class Test Set)

| Metric | Classical 2D CNN Baseline | Hybrid Quantum CNN (Quanv) |
| :--- | :---: | :---: |
| **Accuracy** | 83.84% | **81.44%** |
| **ROC-AUC (One-vs-Rest)** | 0.9778 | **0.9772** |
| **Weighted F1-Score** | 0.8361 | **0.8113** |
| **Macro F1-Score** | 0.8126 | **0.7732** |

*Note: The Quantum CNN achieves competitive classification confidence (0.9772 ROC-AUC) using static quantum entangling circuits without backpropagating parameter-shift gradients into the quantum layer.*

---

## Repository Structure

```text
.
├── data/                  # BloodMNIST dataset store
├── models/
│   ├── classical_cnn.py   # Baseline 2D Classical CNN
│   └── quanv_cnn.py       # 4-Qubit Qiskit Filter + PyTorch Classifier
├── utils/
│   ├── dataset.py         # Grayscale data loading & transforms
│   └── metrics.py         # Multi-class F1, ROC-AUC & Confusion Matrix visualizer
├── outputs/               # Confusion matrix comparison plots
├── preprocess_quantum.py  # Offline quantum feature extraction script
├── train.py               # Master training & evaluation script
├── requirements.txt       # Dependencies
└── README.md
