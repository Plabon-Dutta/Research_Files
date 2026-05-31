# Privacy-Preserving Bengali Sign Language Recognition using Federated Learning

Official implementation accompanying the paper:

**Privacy-Preserving Bengali Sign Language Recognition Using Federated Learning**

---

## Overview

This repository contains the source code used in our research on **privacy-preserving Bengali Sign Language (BSL) recognition using Federated Learning**.

Traditional sign language recognition systems require collecting user data on a centralized server, which introduces significant privacy concerns. To address this issue, we propose a Federated Learning (FL) framework that enables collaborative model training while keeping user data on local devices.

The proposed framework utilizes multiple deep learning architectures combined with different federated aggregation algorithms to classify Bengali Sign Language gestures from the BDSL-47 dataset.

---

## Key Features

* Federated Learning-based Bengali Sign Language Recognition
* Privacy-preserving decentralized training
* Flower Federated Learning Framework
* Multiple aggregation algorithms

  * FedAvg
  * FedProx
  * FedOpt
* Multiple deep learning architectures
* Benchmark comparison with state-of-the-art approaches

---

## Dataset

The experiments were conducted using the **BDSL-47 Dataset**.

Dataset characteristics:

* Total Images: 47,000
* Number of Classes: 47
* Bengali Alphabets: 37
* Bengali Numerals: 10
* Images per Class: 1,000
* Participants: 10 users

The dataset contains Bengali sign language gestures represented as RGB images and landmark-based features extracted using MediaPipe.

---

## Repository Structure

```text
FinalCodes/
│
├── FedAVG/
│   └── Federated Averaging implementation
│
├── FedOPT/
│   └── Adaptive Federated Optimization implementation
│
├── FedProx/
│   └── Federated Proximal Learning implementation
│
├── Flower/
│   └── Flower-based Federated Learning framework
│
├── Scripts/
│   └── Supporting utility scripts
│
└── README.md
```

---

## Federated Learning Framework

The proposed architecture follows a standard Federated Learning workflow:

1. The global server initializes a model.
2. Local clients receive the global model.
3. Each client trains on local sign language data.
4. Model parameters are sent back to the server.
5. The server aggregates updates using a federated aggregation algorithm.
6. The updated global model is redistributed.
7. The process repeats until convergence.

This approach ensures that raw user data never leaves the local device.

---

## Aggregation Algorithms

### FedAvg

Federated Averaging combines model updates from multiple clients through weighted averaging and updates the global model after each communication round.

### FedProx

FedProx extends FedAvg by introducing a proximal regularization term that helps address data heterogeneity among clients.

### FedOpt

FedOpt employs adaptive federated optimization techniques to improve convergence and model performance in decentralized environments.

---

## Deep Learning Models Evaluated

The research evaluates several state-of-the-art architectures:

* CNN
* VGG16
* VGG19
* InceptionV3
* MobileNetV2
* ResNet50

These models were trained and evaluated using different federated aggregation algorithms.

---

## Experimental Configuration

Hardware:

* Intel Core i7 (13th Generation)
* 64 GB RAM
* NVIDIA RTX 4080 (16 GB)
* SSD + HDD Storage

Software:

* Python
* TensorFlow
* Keras
* Scikit-Learn
* OpenCV
* Flower Federated Learning Framework

Experimental setup from the paper.

---

## Hyperparameters

| Parameter     | Value                     |
| ------------- | ------------------------- |
| Learning Rate | 0.001                     |
| Epochs        | 20                        |
| Batch Size    | 64                        |
| Image Size    | 128 × 128                 |
| Loss Function | Categorical Cross-Entropy |
| Optimizer     | Adam                      |

The same configuration was used for most evaluated models.

---

## Results

The proposed federated framework achieved strong classification performance while preserving user privacy.

### Best Performing Models

| Algorithm | Model | Test Accuracy |
| --------- | ----- | ------------- |
| FedAvg    | VGG19 | 98.36%        |
| FedAvg    | VGG16 | 98.25%        |
| FedProx   | VGG19 | 97.59%        |
| FedOpt    | VGG16 | 97.90%        |

The highest test accuracy was achieved using **VGG19 with FedAvg**, reaching **98.36% accuracy** on the BDSL-47 dataset.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Plabon-Dutta/ResearchFiles.git

cd ResearchFiles/FinalCodes
```

Install required dependencies

---

## Running Experiments

Navigate to the desired aggregation method directory:

```bash
cd FedAVG
```

or

```bash
cd FedProx
```

or

```bash
cd FedOPT
```

Then execute the corresponding training script.

---

## Privacy Considerations

The proposed framework preserves privacy by:

* Keeping user data local
* Sharing only model parameters
* Avoiding centralized storage of sensitive sign language data
* Utilizing decentralized collaborative training through Flower Federated Learning

This makes the system suitable for privacy-sensitive computer vision applications.

---

## Citation

If you use this repository or our research, please cite:

```bibtex
@article{diba2024privacy,
  title={Privacy-Preserving Bengali Sign Language Recognition Using Federated Learning},
  author={Diba, Bidita Sarkar and Plabon, Jayonto Dutta and others},
  journal={Engineering Applications of Artificial Intelligence},
  volume={134},
  pages={108657},
  year={2024},
  publisher={Elsevier}
}
```

---

## Authors

* Bidita Sarkar Diba
* Jayonto Dutta Plabon
* Co-authors of the original publication

---

## License

This repository is intended for academic and research purposes.

Please cite the associated publication when using the code or methodology.

---

## Acknowledgements

We thank the contributors of:

* Flower Federated Learning Framework
* TensorFlow
* Keras
* BDSL-47 Dataset

for supporting research in privacy-preserving Bengali Sign Language recognition.
