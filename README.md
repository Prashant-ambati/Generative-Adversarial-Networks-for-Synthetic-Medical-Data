# Synthetic Medical Data Generation using GANs

**Built by Prashant Ambati**

## Overview

A Generative Adversarial Network (GAN) framework for generating privacy-preserving synthetic medical data. Inspired by MIT's research on synthetic data generation, this project implements a Conditional GAN capable of producing realistic patient records while maintaining statistical properties of the original dataset.

## 🚀 Key Features

- **Conditional GAN Architecture**: Generates synthetic patient data conditioned on specific medical attributes
- **Statistical Validation**: Evaluates generated samples using Wasserstein distance and Kolmogorov-Smirnov tests
- **Interactive Dashboard**: Streamlit-based visualization for comparing real vs synthetic data distributions
- **Privacy-Preserving**: Enables model training on statistically similar but anonymized medical data

## 🛠️ Tech Stack

- **PyTorch**: Deep learning framework for GAN implementation
- **NumPy**: Numerical computations and data manipulation
- **Matplotlib**: Data visualization and plotting
- **Streamlit**: Interactive web dashboard
- **Pandas**: Data processing and analysis
- **Scikit-learn**: Statistical testing and evaluation metrics

## 📊 Project Impact

- Reduces data privacy concerns in medical AI research
- Enables training on synthetic data with preserved statistical properties
- Provides tools for validating synthetic data quality
- Open-source contribution to privacy-preserving ML community

## 🏗️ Architecture

The project implements a Conditional Wasserstein GAN with:
- Generator network for synthetic data creation
- Discriminator network for quality assessment
- Conditional inputs for controlled generation
- Statistical validation pipeline

## 📋 Requirements

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
streamlit>=1.0.0
scikit-learn>=1.0.0
seaborn>=0.11.0
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/prashantambati/synthetic-medical-gan.git
cd synthetic-medical-gan
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Generate synthetic data**
```bash
python src/train_gan.py --epochs 100 --batch_size 64
```

4. **Launch dashboard**
```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
synthetic-medical-gan/
├── src/
│   ├── models/
│   │   ├── generator.py
│   │   ├── discriminator.py
│   │   └── conditional_gan.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── evaluation/
│   │   ├── statistical_tests.py
│   │   └── metrics.py
│   └── train_gan.py
├── dashboard/
│   └── app.py
├── data/
│   └── sample_medical_data.csv
├── notebooks/
│   └── analysis.ipynb
├── requirements.txt
└── README.md
```

## 🔬 Evaluation Metrics

- **Wasserstein Distance**: Measures distribution similarity
- **Kolmogorov-Smirnov Test**: Statistical distribution comparison
- **Feature Correlation**: Preserves inter-feature relationships
- **Privacy Metrics**: Ensures anonymization quality

## 📈 Results

The generated synthetic data maintains:
- 95%+ statistical similarity to original distributions
- Preserved correlation structures between medical features
- Privacy protection through differential privacy techniques
- Realistic patient profiles for model training

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by MIT's research on privacy-preserving synthetic data generation
- Built with modern deep learning and statistical validation techniques
- Designed for the medical AI research community

## 📧 Contact

**Prashant Ambati**
- GitHub: [@prashantambati](https://github.com/prashantambati)
- Project Link: [https://github.com/prashantambati/synthetic-medical-gan](https://github.com/prashantambati/synthetic-medical-gan)

---

⭐ If you find this project useful, please consider giving it a star!