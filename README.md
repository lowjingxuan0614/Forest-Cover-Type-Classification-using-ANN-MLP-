# Forest-Cover-Type-Classification-using-ANN-MLP-
This project implements an end-to-end machine learning pipeline for forest cover type classification using the UCI Covertype dataset. The workflow includes exploratory data analysis (EDA), feature selection, outlier removal, ANN model building, hyperparameter tuning (SGD vs Adam), ensemble learning, and model persistence.  

---

## 📘 Access the Notebook

You can open the notebook directly via Google Colab:
🔗 *[https://colab.research.google.com/drive/1Jl4OlPzpFpXrTXeRoipa7O3_-d51ljsP?usp=sharing]*

Alternatively, download the `.ipynb` file and open it locally using Jupyter Notebook or upload it to Google Colab.

---

## 📊 Dataset

* **Dataset**: UCI Forest Cover Type
* **Source**: `ucimlrepo` (dataset ID: 31)
* **Task**: Multi-class classification (7 forest cover types)
* **Features**:

  * Numerical features (elevation, slope, distances, etc.)
  * One-hot encoded categorical features (wilderness areas, soil types)

The dataset is fetched automatically using:

```python
from ucimlrepo import fetch_ucirepo
```

No manual download is required.

---

## 🧪 Project Workflow

### 1. Exploratory Data Analysis (EDA)

* Load and inspect dataset
* Check variable types
* Verify missing values
* Visualize numerical feature distributions
* Analyze feature–target correlations

### 2. Data Preprocessing

* Split numerical and categorical features
* Outlier detection using **Isolation Forest**
* Remove detected outliers
* Feature selection based on correlation ranking
* Scale numerical features using **StandardScaler**
* Train–test split

---

## 🧠 Model Architecture (ANN / MLP)

Baseline neural network architecture:

* Input layer
* 3 hidden layers:

  * 256 neurons (ReLU)
  * 128 neurons (ReLU)
  * 64 neurons (ReLU)
* Output layer: 8 neurons (Softmax)
* Loss: `sparse_categorical_crossentropy`

---

## ⚙️ Model Training & Tuning

### Baseline Model

* Optimizer: Adam
* Batch size: 32
* Early stopping enabled
* Achieved ~93% accuracy

### Hyperparameter Tuning (100k subset)

* **Optimizers**: SGD and Adam
* Tuned parameters:

  * Learning rate
  * Batch size
  * Number of epochs
* Validation accuracy used for model selection

---

## 📈 Model Comparison

The following models are trained and compared:

* Baseline MLP
* Tuned SGD model
* Tuned Adam model

Evaluation includes:

* Validation accuracy
* Training time
* Accuracy and loss curves
* Classification report
* Confusion matrix

---

## 🗳️ Ensemble Learning

Two ensemble strategies are implemented:

* **Hard Voting**: Majority vote across models
* **Soft Voting**: Average predicted probabilities

Performance comparison is conducted against individual models.

---

## 💾 Model Saving & Loading

All trained models are saved to Google Drive:

```text
/ML_model/
├── baseline_model.h5
├── sgd_best_model.h5
└── adam_best_model.h5
```

Models are later reloaded for inference and ensemble prediction.

---

## 📊 Evaluation Metrics

* Accuracy
* Classification report
* Confusion matrix (best individual model)
* Comparison between baseline, tuned models, and ensembles

---

## 🛠️ Technologies Used

* Python
* Google Colab
* NumPy, Pandas
* Scikit-learn
* TensorFlow / Keras
* Matplotlib, Seaborn
* UCI ML Repo API

---

## 📌 Notes

* Numerical features are scaled; categorical features remain one-hot encoded
* Outlier removal improves model stability and performance
* Adam optimizer generally converges faster than SGD in this experiment

---

