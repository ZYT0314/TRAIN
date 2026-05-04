
---

## 📑 Module Descriptions

### 1. `Data pre-processing/`
This directory contains comprehensive code for handling missing values and data preprocessing.
- Implements **multiple imputation** methods for robust missing value estimation.
- Includes **KNN-based imputation** for local structure-aware missing data filling.
- Prepares clean, standardized datasets for subsequent modeling workflows.

---

### 2. `Function/`
This module provides foundational utilities and baseline model implementations:
- Generates synthetic datasets based on real-world functions for controlled testing.
- Contains plotting and visualization code for data distribution and model behavior.
- Implements fitting capability assessments for both baseline models and the TRAIN model, enabling performance comparisons across different approaches.

---

### 3. `Multi-omics Fusion/`
This directory includes pre-training code for models leveraging multi-omics data integration:
- Implements multi-omics fusion strategies to combine information from different data modalities.
- Provides pre-training pipelines for models designed to leverage cross-omic relationships.
- Facilitates the development of more robust and predictive models by integrating complementary biological signals.

---

### 4. `TRAIN/`
This directory contains complete pre-training code for single-omics data modalities:
- Implements the full pre-training pipeline of the TRAIN model for each individual omics type.
- Supports training workflows optimized for single-omics datasets, including feature engineering and model fitting.
- Serves as the core implementation of the TRAIN model’s single-modality learning capabilities.

---

### 5. `TRAIN_Validation/`
This directory provides code for external validation of the TRAIN model’s generalization performance:
- Uses independent, external datasets to evaluate the model’s performance on unseen data.
- Implements validation workflows to assess the TRAIN model’s ability to generalize beyond the training set.
- Includes metrics and visualizations to quantify and report the model’s robustness across different datasets.

---

### 6. `TRAIN_main/SHAP/`
This directory contains code for model interpretability analysis using SHAP:
- Implements SHAP (SHapley Additive exPlanations) functions to compute feature importance and model behavior.
- Provides visualization code for generating SHAP summary plots, force plots, and dependency plots.
- Enables detailed interpretability of the TRAIN model’s predictions, helping to understand the contributions of individual features.

---

*(Note: The `TRAIN_Follow` module, containing long-term steady-state prediction code for single-omics data, will be added to the repository in subsequent updates.)*
