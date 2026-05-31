# EmojiSculpt: Federated Learning-Based Architecture for Personalized Next Emoji Prediction

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)]()
[![Python](https://img.shields.io/badge/Python-3.9+-green)]()
[![License](https://img.shields.io/badge/License-Academic-orange)]()

Official implementation accompanying the IEEE Access paper:

**Federated Learning-Based Architecture for Personalized Next Emoji Prediction for Social Media Comments**

---

## 📖 Overview

Emoji prediction has emerged as an important Natural Language Processing (NLP) task due to the widespread use of emojis in online communication.

This repository contains the implementation and experimental notebooks used in our IEEE Access publication, where we introduce a privacy-preserving Federated Learning framework for Bengali next-emoji prediction using transformer-based language models.

The proposed architecture:

- Preserves user privacy through Federated Learning
- Utilizes transformer-based language models
- Predicts emojis from Bengali social media comments
- Compares multiple BERT-family architectures
- Evaluates FedAvg and FedProx aggregation strategies

---

## 🚀 Features

- Bengali emoji prediction
- Federated Learning framework
- Privacy-preserving training
- Transformer-based architectures
- Comparative model evaluation
- Reproducible experiments

---

## 🤖 Evaluated Models

The following transformer architectures were evaluated:

| Model | Notebook |
|---------|---------|
| Bangla BERT | `bangla-bert-base.ipynb` |
| BERT Base | `bert_base.ipynb` |
| BERT Base Variant | `bert_base_1.ipynb` |
| DistilBERT Base | `distilbert-base.ipynb` |
| Electra Small Discriminator | `electra-small-discriminator.ipynb` |
| ALBERT Base V2 | `albert-base-v2.ipynb` |
| XLM-RoBERTa Base | `xlm-roberta-base.ipynb` |

---

## 📊 Dataset

The experiments were conducted on Bengali social media comments collected through web scraping.

### Dataset Statistics

- Total collected comments: 100,000+
- Filtered Bengali comments: 5,000
- Emoji classes: 20
- Emotion categories:
  - Happy
  - Sad
  - Love
  - Normal

---

## ⚙️ Federated Learning Configuration

### FedAvg

Standard Federated Averaging algorithm.

### FedProx

Federated optimization with proximal regularization.

### Hyperparameters

| Parameter | Value |
|------------|---------|
| Learning Rate | 1e-5 |
| Epochs | 2 |
| Communication Rounds | 10 |
| Number of Clients | 5 |
| Optimizer | AdamW |

---

## 📈 Results

The proposed framework achieved up to:

| Model | Algorithm | Accuracy |
|---------|------------|------------|
| BanglaHateBERT | FedAvg | 98.36% |
| Multilingual BERT | FedAvg | 98.00% |
| BanglaHateBERT | FedProx | 97.00% |
| Multilingual BERT | FedProx | 98.00% |

The results demonstrate that privacy-preserving federated learning can achieve highly competitive emoji prediction performance while protecting user data.

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/EmojiSculpt-FL.git

cd EmojiSculpt-FL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

---

## 📜 Citation

If you use this repository in your research, please cite:

```bibtex
@ARTICLE{10663425,
  author={Mistry, Durjoy and Plabon, Jayonto Dutta and Diba, Bidita Sarkar and Mukta, Md Saddam Hossain and Mridha, M. F.},
  journal={IEEE Access},
  title={Federated Learning-Based Architecture for Personalized Next Emoji Prediction for Social Media Comments},
  year={2024},
  volume={12},
  pages={140339-140358},
  doi={10.1109/ACCESS.2024.3448470}
}
```

---

## 📝 Paper

IEEE Access (2024)

Title:

**Federated Learning-Based Architecture for Personalized Next Emoji Prediction for Social Media Comments**

DOI:

https://doi.org/10.1109/ACCESS.2024.3448470

---

## 👥 Authors

- Durjoy Mistry
- Jayonto Dutta Plabon
- Bidita Sarkar Diba
- Md Saddam Hossain Mukta
- M. F. Mridha

---

## ⭐ Acknowledgement

If you find this repository useful, please consider starring the repository and citing our paper.
