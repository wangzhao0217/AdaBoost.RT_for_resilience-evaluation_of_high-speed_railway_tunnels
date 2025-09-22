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
# Run the data generation script first
exec(open('Randomly generate 200 numerical simulation datasets.py').read())

# Then run the main AdaBoost.RT analysis
exec(open('AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py').read())

# The scripts will generate:
# - ground_motion_cartesian_200.txt (200 samples)
# - functionality_Q_t.csv (functionality curves)
# - fragility_*.csv (fragility curves for each damage state)
```

### Custom Analysis

```python
# Run the main script to load all classes and functions
exec(open('AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py').read())

import numpy as np
import pandas as pd

# Load sample data
df = pd.read_csv('sample_data.txt', sep='\t')
X = df[['M', 'D', 'PGA']].values
y = df['DI'].values

# Create custom AdaBoost.RT configuration
from sklearn.svm import SVR
cfg = AdaBoostRTConfig(
    base_estimator=SVR(kernel='rbf', C=1.0, epsilon=0.01),
    n_estimators=100,
    phi=0.1,
    learning_rate=1.0,
    random_state=42
)

# Initialize and train model
model = AdaBoostRT(cfg)
# model.fit(X, y)  # Use your training data
# predictions = model.predict(X_test)
```

## 📁 Repository Structure

```
tunnel-seismic-resilience/
├── README.md
├── AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py  # Main analysis script
├── Randomly generate 200 numerical simulation datasets.py                   # Data generation script
├── sample_data.txt                                                         # Sample dataset (20 records)
└── generated_data/                                                         # Generated by scripts
    ├── ground_motion_cartesian_200.txt                                     # Full 200-record dataset
    ├── functionality_Q_t.csv                                              # Functionality curves
    └── fragility_*.csv                                                     # Fragility curves per damage state
```

## 📊 Data Generation and Description

### Numerical Simulation Dataset Generation

**Randomly generate 200 numerical simulation datasets** (`Randomly generate 200 numerical simulation datasets.py`): This script generates a comprehensive dataset using Cartesian product rules to avoid extensive numerical simulations. It systematically combines 5 ground motion types with 40 PGA levels (5×40=200 samples) to create a complete parameter space coverage. The script also includes synthetic target generation using a Random Forest regressor for validation purposes.

**AdaBoost.RT for resilience evaluation of high-speed railway tunnels** (`AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py`): The main analysis script implements the AdaBoost.RT algorithm using the generated data. It processes input features (M, D, PGA) to predict Damage Index (DI) values, generates fragility curves for different damage states, and performs K-fold cross-validation to ensure model reliability and generalization capability. The sample_data.txt file contains representative results from this analysis.

**Note**: Data will be released separately and made available through GitHub Releases.

### Input Parameters Dataset

The dataset includes 200 samples generated using a Cartesian product approach with the following structure:

**Ground Motion Parameters:**
- **GM_Type**: Ground motion type (categorical)
  - Imperial-Vally-06
  - Imperial-Vally-060
  - Landers
  - Menyuan
  - Wenchuan
- **PGA_g**: Peak Ground Acceleration in g-units (0.1g to 1.2g, 40 levels)

**Generated Features in Sample Data:**
- **M**: Moment capacity (kN·m) - Range: 215.7 - 499.8
- **D**: Displacement (mm) - Range: 1.08 - 4.88
- **PGA**: Peak Ground Acceleration (g) - Range: 0.19 - 1.12
- **DI**: Damage Index - Range: 71.5 - 281.5

### Output Variables

- **Damage Index (DI)**: Primary output variable representing tunnel structural damage
  - Calculated based on moment capacity, displacement, and PGA relationships
  - Used for fragility analysis and resilience assessment
- **Fragility Parameters**: Probability of exceedance for different damage states
  - Slight damage (DI > 1.0)
  - Moderate damage (DI > 1.5)
  - Extensive damage (DI > 2.5)
  - Complete damage (DI > 3.5)

### Data Availability

The research datasets are available in the following formats:

**Current Available Data:**
- `sample_data.txt`: 20 sample records with M, D, PGA, and DI values (tab-separated format)
- `ground_motion_cartesian_200.txt`: Full 200-record dataset generated using Cartesian product (generated by data generation script)

**Data Generation Scripts:**
- `Randomly generate 200 numerical simulation datasets.py`: Creates comprehensive dataset using Cartesian product approach
- `AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py`: Main analysis script with model implementation

**Additional Data (Coming Soon):**
- `simulation_results.txt`: FEM simulation results
- `daliang_tunnel_validation/`: Real earthquake damage validation data

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

- **Lianjie Yang** - College of Environment and Civil Engineering, Chengdu University of Technology
- **Chunlei Xin** - State Key Laboratory of Geohazard Prevention and Geoenvironment Protection, Chengdu University of Technology 
- **Zhao Wang** (*Corresponding Author*) - Institute for Transport Studies, University of Leeds
- **Xinyuan Yu** - School of Mechanical, Aerospace and Civil Engineering, The University of Sheffield
- **Iman Hajirasouliha** - School of Mechanical, Aerospace and Civil Engineering, The University of Sheffield
- **Wenkai Feng** - State Key Laboratory of Geohazard Prevention and Geoenvironment Protection, Chengdu University of Technology

**Corresponding Author**: z.wang13@leeds.ac.uk

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
- **Lianjie Yang**: yanglianjie0804@163.com

---

## 📈 Release History

### v1.0.0 (Current)
- Initial release
- AdaBoost.RT algorithm implementation (`AdaBoost.RT for resilience evaluation of high-speed railway tunnels.py`)
- Data generation script (`Randomly generate 200 numerical simulation datasets.py`)
- Sample dataset with 20 records (`sample_data.txt`)
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
