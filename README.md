# Multilayer Perceptron

## Purpose of the project
The goal of this project is to have a first approach to artificial neural networks and implement the heart of the process. For that, I use the Wisconsin Diagnostic Breast Cancer dataset and I will try to predict if a given patient's tumor is benign or malignant.

Let's first explain the project as a whole and then dive into the mathematical foundations behind it.

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

As illustrated in the pairplot above, malignant tumors (`M`) typically exhibit larger cell nuclei (`radius`, `perimeter`, `area`) and greater contour irregularity (`concavity`, `concave_points`) compared to benign tumors (`B`). While the two classes form distinct clusters, the overlapping regions justify using a **Multilayer Perceptron** to model non-linear decision boundaries:

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

## Installation & Usage

### Prerequisites

This project uses [uv](https://github.com/astral-sh/uv) for dependency management (Python 3.10+).

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jana1738/multilayer_perceptron.git
   cd multilayer_perceptron
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

### Running the Pipeline

1. **Split the dataset** into training (80%) and validation (20%) sets:
   ```bash
   uv run src/separate.py datasets/data.csv
   ```

2. **Train the model** (with early stopping, metrics logging, and learning curves):
   ```bash
   uv run src/train.py datasets/data_training.csv datasets/data_validation.csv
   ```

   *To use Adam, you have to give the string "adam" as second parameter to the denseLayer class method update() and adjust your learning rate.* 

3. **Run predictions** on unseen data using the saved model weights:
   ```bash
   uv run src/predict.py datasets/data_validation.csv
   ```

## Results

Configured with a **`30 -> 24 -> 16 -> 2`** architecture (30 input features, two hidden layers of 24 and 16 neurons, and 2 output classes) and trained using **Adam** (`lr = 0.001`, `batch_size = 32`) with Early Stopping (`patience = 10`), the model converges smoothly without overfitting, reaching its lowest validation loss at **epoch 42**.

### Performance Metrics (Validation Set — 115 samples)

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Best Validation Loss** | `0.0534` | Categorical Cross-Entropy (Epoch 42) |
| **Training Accuracy** | `99.12%` | 450 / 454 correct predictions |
| **Validation Accuracy** | `99.13%` | 114 / 115 correct predictions |
| **Validation Precision** | `100.00%` | Zero false positives |
| **Validation Recall** | `97.67%` | 42 / 43 malignant cases detected |
| **Validation F1-Score** | `0.9882` | Harmonic mean of precision and recall |

### Learning Curves

**Without Adam (Standard SGD):**
<img width="1382" height="481" alt="Training and Validation Loss and Accuracy Curves without Adam" src="https://github.com/user-attachments/assets/261d4a61-383b-4421-9533-d27c80c1f135" />

**With Adam (`lr = 0.001`):**
<img width="1384" height="484" alt="Training and Validation Loss and Accuracy Curves with Adam" src="https://github.com/user-attachments/assets/b1d4a0bb-54c3-4f63-9bf5-16e83ba90123" />

* **Loss & Accuracy Alignment**: Both training and validation curves decrease and plateau together, confirming strong generalization on unseen patient data.
* **Clinical Relevance**: Achieving **100% precision** alongside **97.67% recall** minimizes both false alarms and missed malignant diagnoses.

