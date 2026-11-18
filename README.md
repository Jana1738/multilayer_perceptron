# Multilayer Perceptron

## Purpose of the project
The goal of this project is to have a first approach to artificial neural networks and implement the heart of the process. For that, we use the Wisconsin Diagnostic Breast Cancer dataset and we will try to predict if a given patient's tumor is benign or malignant.

## Description of the dataset

<img width="1900" height="955" alt="Screenshot From 2026-11-18 17-29-34" src="https://github.com/user-attachments/assets/e148a751-44c7-41c3-a075-af2b6b71d5bb" />

*(Not all features are shown in this image)*

First of all, we have to understand what the dataset describes. The **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset contains **569 samples** computed from digitized images of a fine needle aspirate (FNA) of a breast mass, describing the physical characteristics of the cell nuclei present in the image.

### Structure & Features
After dropping the patient ID column, each sample consists of a target diagnosis and **30 real-valued input features**:

* **Target (`diagnosis`)**: Binary classification label indicating whether the mass is **Benign (`B`)** (357 samples, 62.7%) or **Malignant (`M`)** (212 samples, 37.3%).
* **10 Core Cell Nucleus Measurements**: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension.
For more information, see : https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names

For each of these 10 characteristics, three statistical values are recorded per image, resulting in **30 features** ($10 \times 3$):
* **`_mean`**: The mean value across all visible nuclei.
* **`_se`**: The standard error of the measurements.
* **`_worst`**: The mean of the three largest (most severe) values.

As illustrated in the pairplot above, malignant tumors (`M`) typically exhibit larger cell nuclei (`radius`, `perimeter`, `area`) and greater contour irregularity (`concavity`, `concave_points`) compared to benign tumors (`B`). While the two classes form distinct clusters, the overlapping regions justify using a **Multilayer Perceptron** to model non-linear decision boundaries.
