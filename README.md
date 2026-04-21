# 🛡️ ChurnGuard AI: Customer Churn Prediction

ChurnGuard AI is a high-performance **Artificial Neural Network (ANN)** project designed to predict the likelihood of bank customer attrition. It transforms raw demographic and financial data into actionable insights through a sleek, interactive web dashboard.

## 🚀 Key Features
- **Deep Learning Core**: Built with TensorFlow/Keras for high-accuracy classification.
- **Interactive UI**: A premium Streamlit dashboard for real-time predictions.
- **Automated Preprocessing**: Robust pipeline for Label Encoding, One-Hot Encoding, and Standard Scaling.
- **Training Insights**: Integrated with TensorBoard for monitoring model convergence and performance.
- **Early Stopping**: Prevents overfitting by monitoring validation loss during training.

## 🛠️ Technology Stack
| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Deep Learning** | TensorFlow 2.11+, Keras |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **Visualization** | TensorBoard, Matplotlib, Seaborn |
| **Web Framework** | Streamlit |

## 📈 Project Workflow
1. **Data Preprocessing**:
   - Dropped irrelevant features (RowNumber, CustomerId, Surname).
   - Multi-step encoding: Label Encoding for Gender and One-Hot Encoding for Geography.
   - Standardized features using `StandardScaler` to optimize Neural Network performance.
2. **Model Architecture**:
   - **Input Layer**: Matches the feature dimensions (12 inputs).
   - **Hidden Layers**: Two Dense layers (64 and 32 neurons) with ReLU activation.
   - **Output Layer**: Single neuron with Sigmoid activation for binary classification.
3. **Training & Callbacks**:
   - Optimized with **Adam** (learning_rate=0.01).
   - Loss function: **Binary Crossentropy**.
   - Integrated **EarlyStopping** and **TensorBoard** for efficient training.

## 💻 Installation & Usage

### 1. Environment Setup
```powershell
# Create a virtual environment
python -m venv venv

# Activate the environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web App
```powershell
.\venv\Scripts\streamlit run app.py
```

### 3. Monitor Training (TensorBoard)
```powershell
.\venv\Scripts\tensorboard --logdir log
```

## 📁 Project Structure
- `app.py`: Streamlit application file.
- `experiments.ipynb`: Data exploration and model prototyping.
- `model.h5`: The trained ANN model.
- `*.pkl`: Scalers and Encoders used for preprocessing.
- `Churn_Modelling.csv`: The primary dataset.

---
Developed as part of an Advanced ANN Classification series. 📊
