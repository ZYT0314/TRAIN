# TRAIN: Rheumatoid Arthritis Intelligence Network

## Overview

TRAIN is a rheumatology-specific pretrained framework developed for structured biomedical data analysis. The framework integrates large-scale clinical records, molecular multi-omics datasets, and radiology-report-derived phenotypes to support multiple clinically relevant tasks, including joint phenotyping, diagnostic differentiation, and longitudinal rheumatoid arthritis management.

This repository provides the implementation of data preprocessing, model training, multi-omics integration, external validation, cross-disease classification, and model interpretation analyses used in this study.

---

# Repository Structure

## 1. Data pre-process/

This directory contains preprocessing pipelines for preparing heterogeneous biomedical datasets before model development.

Main functions include:

- Data cleaning and normalization.
- Missing-value handling and imputation strategies.
- Feature transformation and dataset preparation.
- Generation of standardized input matrices for subsequent TRAIN model training and evaluation.

These preprocessing workflows ensure consistent representation across clinical, radiological, and molecular datasets.

---

## 2. Function/

This directory contains general utility functions and benchmarking components used throughout the study.

Main functions include:

- Generation of synthetic datasets for controlled experiments.
- General model fitting utilities.
- Visualization functions for model performance assessment.
- Benchmarking workflows comparing TRAIN with conventional machine-learning approaches.

This module supports methodological evaluation and reproducibility analyses.

---

## 3. Multi-omics Fusion/

This directory contains the implementation of multi-omics integration and fusion strategies.

Main functions include:

- Integration of transcriptomic and proteomic information.
- Construction of multi-modal feature representations.
- Multi-omics pretraining workflows.
- Evaluation of complementary biological information from different molecular modalities.

This module enables TRAIN to capture shared representations across multiple molecular layers.

---

## 4. TRAIN_Image report/

This directory contains code for radiology-report-based joint phenotype modelling.

Main functions include:

- Processing of structured radiology-derived features.
- Training and evaluation of inflammatory and degenerative joint phenotype classification models.
- Geographic external validation across independent radiology centres.
- Performance evaluation and visualization.

This module supports assessment of TRAIN for imaging-associated clinical phenotyping tasks.

---

## 5. TRAIN_classification/

This directory contains classification workflows for clinical diagnostic differentiation tasks.

Main functions include:

- Model training for disease classification.
- Rheumatic disease differentiation analyses.
- Prediction workflows using clinical and molecular features.
- Evaluation using internal and external cohorts.

This module supports diagnostic classification analyses among rheumatological conditions.

---

## 6. TRAIN_cross classification/

This directory contains cross-condition classification analyses evaluating model transferability.

Main functions include:

- Cross-disease prediction workflows.
- Evaluation of representation transferability across related diseases.
- Assessment of TRAIN performance under different classification scenarios.

This module investigates whether pretrained representations can support broader rheumatological discrimination tasks.

---

## 7. TRAIN_Validation/

This directory contains external validation workflows for assessing TRAIN generalizability.

Main functions include:

- Application of pretrained models to independent cohorts.
- Evaluation on unseen datasets.
- Calculation of performance metrics.
- Generation of validation results and visualization.

This module supports assessment of model robustness and transportability across institutions.

---

## 8. TRAIN_SHAP/

This directory contains model interpretation analyses based on SHAP (SHapley Additive exPlanations).

Main functions include:

- Calculation of feature contribution scores.
- Identification of important predictive variables.
- Generation of SHAP summary plots and feature interpretation analyses.
- Exploration of model decision patterns.

This module provides interpretability analysis for understanding clinical and molecular contributors to TRAIN predictions.

---

# Reproducibility

All scripts required to reproduce the main analyses reported in the manuscript are provided in this repository.

The repository includes:

- Data preprocessing workflows.
- TRAIN model implementation.
- Multi-omics integration pipelines.
- Clinical classification analyses.
- External validation procedures.
- Model interpretation analyses.

Due to restrictions related to human genetic resources, clinical data governance, and institutional regulations, raw clinical records and in-house molecular datasets are not publicly released.

Requests for controlled access to eligible datasets should be directed to the corresponding authors and will be evaluated according to institutional data-sharing procedures.

---

# Software Requirements

The code was developed using Python-based machine-learning workflows.

Required packages and environment configurations are described within individual modules.

