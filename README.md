# Multilayer Perceptron

## Purpose of the project
The goal of this project is to have a first approach to artificial neural networks and implement the heart of the process. For that, I use the Wisconsin Diagnostic Breast Cancer dataset and I will try to predict if a given patient's tumor is benign or malignant.

Let's first explain the project as a whole and then dive into the mathematical foundation behind it.

## Description of the dataset

<img width="1900" height="955" alt="Dataset Pairplot" src="https://github.com/user-attachments/assets/e148a751-44c7-41c3-a075-af2b6b71d5bb" />

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

As illustrated in the pairplot above, malignant tumors (`M`) typically exhibit larger cell nuclei (`radius`, `perimeter`, `area`) and greater contour irregularity (`concavity`, `concave_points`) compared to benign tumors (`B`). While the two classes form distinct clusters, the overlapping regions justify using a **Multilayer Perceptron** to model non-linear decision boundaries :

<img width="1065" height="739" alt="Non-linear overlapping features pairplot" src="https://github.com/user-attachments/assets/a3f4bb2c-bbc0-468f-adf6-876b2035779c" />

## Data Preprocessing

Before training the model, the original dataset (569 samples) is split directly into two independent subsets using an **80% / 20% ratio**:

* **Training Set (`data_training.csv`)**: **454 samples** (80%) used to compute gradients and update the network's weights and biases.
* **Validation Set (`data_validation.csv`)**: **115 samples** (20%) never seen during the training phase, used to evaluate generalization, tune hyperparameters, and trigger early stopping.

To ensure stable gradient descent and prevent data leakage, all 30 input features are standardized using the mean and standard deviation computed exclusively on the training set.

## Model Architecture & Features

The network is implemented entirely from scratch in Python using **NumPy**, with a modular architecture designed to handle configurable topologies:

* **Input Layer**: 30 neurons corresponding to the standardized tumor features.
* **Hidden Layers**: Fully connected (`Dense`) layers using **ReLU** activation and **He weight initialization** to prevent vanishing or exploding gradients.
* **Output Layer**: 2 neurons with **Softmax** activation to output class probabilities for *Benign* (`B`) and *Malignant* (`M`) predictions.
* **Loss Function**: **Categorical Cross-Entropy** to measure prediction error during training and validation.
* **Optimizers**: Support for standard **Stochastic Gradient Descent (SGD)** as well as **Adam** (combining Momentum and RMSProp with bias correction) for faster and smoother convergence.
* **Regularization & Generalization**: Built-in **Early Stopping** that monitors validation loss and automatically restores the best model weights to prevent overfitting.
* **Evaluation & Visualization**: Tracks loss, accuracy, precision, recall, and F1-score, and generates training and validation history curves using **Matplotlib**.



