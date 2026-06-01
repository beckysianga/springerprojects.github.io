# Sensitivity-Aware Hybrid Differential Privacy for Personalized Federated Learning under Class Imbalance and Data Heterogeneity
This repository contains the implementation of the **Personalized Representation-Aware Adaptive Sensitivity Hybrid Differential Privacy (PR-AS-HDP)** framework for healthcare federated learning.
The framework was developed to address: privacy preservation, class imbalance, data heterogeneity, and privacy–utility trade-offs, in distributed healthcare machine learning environments.

# Overview
The framework combines:
- Personalized Federated Learning (PFL)
- Adaptive Sensitivity Estimation
- Hybrid Differential Privacy (Local + Central DP)
- Adaptive Clipping
- Adaptive Privacy Budgeting
- Rényi Differential Privacy (RDP) Accounting

The framework was evaluated using:
- Cervical Cancer Dataset
- Diabetes Dataset
- Glioma Grading Dataset

# Research Motivation
Healthcare datasets are:
- Highly sensitive
- Distributed across institutions
- Non-IID (Non-Independent and Identically Distributed)
- Frequently class-imbalanced
Traditional centralized machine learning requires collecting patient data into one location, which raises privacy and governance concerns. Federated Learning (FL) addresses this challenge by allowing collaborative model training without sharing raw data. However, model updates can still leak sensitive information. This project proposes a privacy-aware federated learning framework that dynamically adapts privacy protection according to:
- Client behaviour
- Sensitivity dynamics
- Training stability
- Class imbalance
# Main Contributions
## 1. Personalized Federated Learning
- Shared global encoder
- Client-specific local heads
## 2. Adaptive Sensitivity Estimation
- Sensitivity is estimated dynamically instead of using fixed worst-case bounds
## 3. Hybrid Differential Privacy
- Local Gaussian noise at the client side
- Central Gaussian noise at the server side
## 4. Adaptive Privacy Budgeting
- Privacy budgets are allocated dynamically across communication rounds
## 5. Adaptive Clipping
- Clipping bounds change according to update distributions
## 6. RDP Accounting
- Rényi Differential Privacy is used for tighter privacy accounting
# Framework Architecture
| Component | Purpose |
|---|---|
| Shared Global Encoder | Learns transferable global representations |
| Client-Specific Heads | Supports personalization under heterogeneous data |
| Adaptive Clipping | Stabilizes updates dynamically |
| Adaptive Sensitivity Estimation | Calibrates privacy noise using client behaviour |
| Hybrid DP | Combines local and central privacy protection |
| Adaptive Budget Scheduling | Dynamically allocates privacy budgets |
| RDP Accountant | Tracks cumulative privacy loss |

# Datasets Used
## 1. Cervical Cancer Dataset
- Highly imbalanced healthcare dataset
- Used to evaluate unstable sensitivity behaviour and privacy degradation
## 2. Diabetes Dataset
- Moderately heterogeneous dataset
- Used to evaluate privacy–utility stability
## 3. Glioma Grading Dataset
- More stable sensitivity dynamics
- Used to evaluate personalized learning under strong privacy preservation
# Repository Structure 
```text
├── SpringerCervicalCancerDataset.ipynb
├── SpringerDiabetesDataset.ipynb
├── SpringerGliomaDataset.ipynb
├── README.md
├── requirements.txt
└── outputs/# Repository Structure

Key Features of the Implementation
The implementation includes:
•	Multi-seed experimentation
•	Federated client simulation
•	Non-IID Dirichlet partitioning
•	Personalized learning
•	Dynamic clipping
•	Adaptive sensitivity estimation
•	Hybrid differential privacy
•	RDP accounting
•	Early stopping
•	Statistical evaluation
•	Confidence intervals
•	Privacy fairness verification
The codebase supports reproducible experiments across multiple healthcare datasets.
Experimental Configuration
Parameter	Value
Number of Clients	8
Communication Rounds	30
Client Participation	75%
Local Epochs	2
Dirichlet Alpha	0.5
Random Seeds	42, 52, 62
Differential Privacy Delta	1e-5
The implementation uses adaptive clipping and adaptive privacy scheduling during training.
Ablation Studies
Ablation	Description
A1	Adaptive clipping only
A2	Adaptive clipping + adaptive budgeting
A3	Adaptive clipping + adaptive budgeting + adaptive sensitivity
A4	Full PR-AS-HDP framework
The ablation studies help isolate the contribution of each framework component.
Evaluation Metrics
The framework evaluates:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	AUPRC (Area Under Precision Recall Curve)
AUPRC is emphasized because healthcare datasets are often highly imbalanced.
Key Research Findings
The experiments showed that:
•	Privacy–utility behaviour is highly dataset-dependent
•	Stable sensitivity leads to better utility
•	Unstable sensitivity increases privacy noise and degrades performance
•	Adaptive mechanisms improve convergence stability
•	Differential privacy impacts datasets differently
•	Sensitivity dynamics influence performance more than the nominal privacy budget
Running the Codes
Install Dependencies
pip install numpy pandas scikit-learn matplotlib torch scipy ucimlrepo
Run Cervical Cancer Experiments
python cervical_cancer_colab_ready.py
Run Diabetes Experiments
python diabetes_colab_ready.py
Run Glioma Experiments
python glioma_grading_colab_ready.py
Outputs Generated
The framework automatically generates:
•	Validation histories
•	Test performance metrics
•	Privacy ledgers
•	Mean ± standard deviation tables
•	Confidence intervals
•	Privacy fairness statistics
•	AUPRC convergence plots
Privacy Mechanisms
Local Differential Privacy (LDP)
Noise is added locally before client updates are transmitted.
Central Differential Privacy (CDP)
Additional noise is added during server aggregation.
Adaptive Sensitivity
Sensitivity is estimated dynamically using:
•	Representation dispersion
•	Clipped update magnitude
•	Class imbalance
RDP Accounting
Privacy loss is accumulated across communication rounds using Rényi Differential Privacy accounting.
Intended Use
This repository is intended for:
•	Federated learning research
•	Differential privacy research
•	Healthcare AI research
•	Privacy-preserving machine learning
•	Graduate and PhD research
•	Experimental benchmarking
Citation
If you use this repository in your research, please cite:
@article{okaka2026prashdp,
  title={Sensitivity-Aware Hybrid Differential Privacy for Personalized Federated Learning under Class Imbalance and Data Heterogeneity},
  author={Okaka, Rebecca Adhiambo and Karanja, Evanson Mwangi and Oteyo, Isaac Nyabisa},
  year={2026}
}
Author
Rebecca Adhiambo Okaka
Research Areas
•	Federated Learning
•	Differential Privacy
•	Privacy-Preserving AI
•	Healthcare Machine Learning
•	Adaptive Privacy Mechanisms
Acknowledgements
This work builds upon research in:
•	Federated Learning
•	Differential Privacy
•	Adaptive Clipping
•	Personalized Federated Learning
•	Rényi Differential Privacy
•	Healthcare AI Research
Special thanks to the open-source healthcare machine learning and privacy research communities.


