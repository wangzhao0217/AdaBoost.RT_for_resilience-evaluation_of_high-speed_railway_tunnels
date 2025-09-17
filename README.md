# Seismic Resilience Analysis of High-Speed Railway Tunnels Using Ensemble Learning

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxx-blue)](https://doi.org/10.xxxx/xxxx)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)

This repository contains the implementation of an ensemble learning approach for rapid resilience assessment of high-speed railway tunnels crossing strike-slip seismogenic faults, as described in our research paper.

## 📖 Abstract

This study introduces a novel hybrid finite element model to explore the impact of fault dislocation-induced earthquakes on tunnel lining integrity. We developed a rapid resilience assessment model using the **Adaboost.RT algorithm** that evaluates seismic fragility and resilience of tunnels crossing strike-slip faults. The model serves as an effective alternative to traditional nonlinear time-history analysis, providing both high accuracy and computational efficiency.

## 🔬 Research Background

Fault-crossing tunnels are particularly susceptible to severe seismic damage due to fault dislocation and strong seismic motion. Notable examples include:
- **2008 Wenchuan earthquake**: Severe damage to Longdongzi and Longxi Tunnels
- **2022 Menyuan earthquake**: Substantial dislocation damage to Daliang Tunnel
- **2023 Turkey earthquake**: Railway infrastructure damage near Ozan village

This research addresses the critical need for enhanced seismic resilience assessment in active fault zones.

## 🚀 Key Features

- **Ensemble Learning Framework**: Implementation of Adaboost.RT algorithm with multiple base learners (SVR, BP Neural Network, ELM)
- **Rapid Assessment**: Efficient alternative to traditional nonlinear time-history analysis
- **Multi-Parameter Analysis**: Considers tunnel structure, surrounding rock properties, and seismic motion intensity
- **Real-World Validation**: Validated using Daliang Tunnel earthquake damage data
- **Comprehensive Evaluation**: Fragility curves and resilience indices for different tunnel configurations

## 📊 Algorithm Implementation

### Adaboost.RT Algorithm

The core algorithm integrates three base learning models:
- **Support Vector Regression (SVR)**: Robust generalization for small sample sizes
- **Backpropagation Neural Network (BP)**: Adaptive nonlinear mapping capability
- **Extreme Learning Machine (ELM)**: Fast convergence and global optimization

### Key Input Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| PGA | Peak Ground Acceleration | 0.1g - 1.2g |
| Tunnel Structure | Horseshoe, Circular, Shock-absorbing joint | Categorical |
| Shear Modulus | Surrounding rock properties | 1-6 GPa |
| Tunnel Height | Cross-sectional dimensions | 6-12 m |
| Ground Motion Type | Near-fault characteristics | Categorical |

### Performance Metrics

- **Coefficient of Determination (R²)**: 0.959 (test set)
- **Root Mean Square Error (RMSE)**: 0.263
- **Mean Absolute Error (MAE)**: 0.237
- **Maximum Error**: 9.5%

## 🛠️ Installation

### Prerequisites

```bash
python >= 3.7
numpy >= 1.19.0
pandas >= 1.2.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
```

### Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Clone Repository

```bash
git clone https://github.com/yourusername/tunnel-seismic-resilience.git
cd tunnel-seismic-resilience
```

## 📈 Usage

### Quick Start

```python
# Import the main module
exec(open('Establishment of an integrated multi-learning model based on the Adaboost.RT algorithm.py').read())

# Generate sample data and run complete workflow
results, data, model = run_workflow(
    input_path='tunnel_data.txt',
    sim_path='simulation_results.txt',
    n_experiments=10
)
```

### Custom Analysis

```python
# Run the main script to load all classes and functions
exec(open('Establishment of an integrated multi-learning model based on the Adaboost.RT algorithm.py').read())

import numpy as np
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge

# Define base estimators
base_estimators = [
    SVR(kernel='rbf', C=1.0, epsilon=0.1),
    MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000),
    Ridge(alpha=0.01)
]

# Initialize and train model
model = AdaboostRT(
    base_estimators=base_estimators,
    n_estimators=50,
    threshold=0.1,
    learning_rate=0.8
)

# model.fit(X_train, y_train)  # Use your data
# predictions = model.predict(X_test)
```

## 📁 Repository Structure

```
tunnel-seismic-resilience/
├── README.md
├── Establishment of an integrated multi-learning model based on the Adaboost.RT algorithm.py
└── data/                           # [Data to be released separately]
    ├── tunnel_data.txt            # Input parameters dataset (coming soon)
    ├── simulation_results.txt     # FEM simulation results (coming soon)
    └── daliang_tunnel_validation/ # Real earthquake damage data (coming soon)
```

## 📊 Data Generation and Description

### Numerical Simulation Dataset Generation

**Randomly generate 200 numerical simulation datasets**: This code is designed to avoid conducting extensive numerical simulations by using Cartesian product rules to make the input data more comprehensive. The Cartesian product approach systematically combines different parameter values to create a comprehensive dataset that covers the entire parameter space efficiently, reducing computational costs while maintaining data completeness.

**AdaBoost.RT for resilience evaluation of high-speed railway tunnels**: Through numerical simulation results, as shown in the my_data.txt file, the AdaBoost.RT algorithm is used to generate vulnerability curves and functional curve data, and K-fold cross-validation is performed on the data to ensure model reliability and generalization capability.

**Note**: Data will be released separately and made available through GitHub Releases.

### Input Parameters Dataset

The dataset includes 200 samples with the following features:

### Input Parameters Dataset

The dataset will include 200 samples with the following features:

**Seismic Parameters:**
- Peak Ground Acceleration (PGA)
- Peak Ground Velocity (PGV)
- Arias Intensity
- Dominant Period
- Duration
- Ground Motion Type

**Structural Parameters:**
- Lining Thickness
- Concrete Strength
- Reinforcement Ratio
- Tunnel Height
- Cross-sectional Shape

**Geological Parameters:**
- Surrounding Rock Shear Modulus
- Rock Density
- Fault Characteristics

### Output Variables

- **Damage Index (DI)**: Ratio of actual to bearing moment capacity
- **Resilience Index**: Multi-dimensional performance metric
- **Fragility Parameters**: Probability of exceedance for different damage states

### Data Availability

The research datasets will be made available in the following formats:
- `tunnel_data.txt`: Input parameters in tab-separated format
- `simulation_results.txt`: FEM simulation results
- `daliang_tunnel_validation/`: Real earthquake damage validation data

**Coming Soon**: Data will be released as part of the GitHub release packages.

## 🎯 Key Findings

1. **Parameter Importance**: PGA shows strongest correlation with tunnel damage (weight: 0.36)
2. **Structural Influence**: Circular tunnels demonstrate superior seismic performance
3. **Rock Properties**: Higher shear modulus significantly improves tunnel resilience
4. **Height Effects**: Larger tunnel heights increase damage probability
5. **Mitigation Strategies**: Shock-absorbing joints reduce damage by up to 95.6%

## 🏗️ Case Study: Daliang Tunnel

The model was validated using the 2022 Menyuan Ms 6.9 earthquake damage to Daliang Tunnel:

- **Location**: Qinghai Province, China (101.26°E, 37.77°N)
- **Fault**: Lenglongling fault (F5) - left-lateral strike-slip
- **Damage Extent**: 350m severely damaged section (5.33% of total length)
- **Maximum Displacement**: 1.78m horizontal, 0.7m vertical
- **Validation Result**: Model predictions align well with observed damage patterns

## 📚 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{yang2024seismic,
  title={Seismic resilience analysis of high-speed railway tunnels across fault zones using ensemble learning approach},
  author={Yang, Lianjie and Xin, Chunlei and Wang, Zhao and Yu, Xinyuan and Hajirasouliha, Iman and Feng, Wenkai},
  journal={[Journal Name]},
  year={2024},
  doi={10.xxxx/xxxx}
}
```

## 👥 Authors

- **Lianjie Yang** - State Key Laboratory of Geohazard Prevention and Geoenvironment Protection, Chengdu University of Technology
- **Chunlei Xin** - College of Environment and Civil Engineering, Chengdu University of Technology
- **Zhao Wang** (*Corresponding Author*) - Institute for Transport Studies, University of Leeds
- **Xinyuan Yu** - School of Mechanical, Aerospace and Civil Engineering, The University of Sheffield
- **Iman Hajirasouliha** - School of Mechanical, Aerospace and Civil Engineering, The University of Sheffield
- **Wenkai Feng** - State Key Laboratory of Geohazard Prevention and Geoenvironment Protection, Chengdu University of Technology

**Corresponding Author**: z.wang13@leeds.ac.uk

## 🤝 Contributing

We welcome contributions to improve the algorithm and extend its applications. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was supported by:
- National Natural Science Foundation of China (Grant No. 52108361)
- Sichuan Science and Technology Program of China (Grant Nos. 25CXCY0063 and 2024ZYD0154)
- State Key Laboratory of Geohazard Prevention and Geoenvironment Protection Independent Research Project (Grant No. SKLGP2022Z015)

## 📞 Contact

For questions, suggestions, or collaborations, please contact:

- **Zhao Wang**: z.wang13@leeds.ac.uk
- **Chunlei Xin**: xinchunlei@cdut.edu.cn

---

## 📈 Release History

### v1.0.0 (Current)
- Initial release
- Adaboost.RT algorithm implementation (`Establishment of an integrated multi-learning model based on the Adaboost.RT algorithm.py`)
- Complete documentation and README

### Upcoming Releases
- **v1.1.0**: Research datasets release
  - Tunnel input parameters dataset
  - FEM simulation results
  - Daliang Tunnel validation data
- **v1.2.0**: Additional features
  - [ ] Real-time damage assessment module
  - [ ] Extended validation with additional case studies
  - [ ] GUI interface for practitioners

### Data Release Plan
The research data will be made available through GitHub Releases to ensure proper versioning and accessibility. Each dataset will include:
- Detailed documentation
- Format specifications
- Usage examples
- Validation instructions

---

**Keywords**: Tunnel engineering, Lining structure, Strike-slip seismic fault, Ensemble learning, Seismic resilience, Machine learning, Earthquake engineering