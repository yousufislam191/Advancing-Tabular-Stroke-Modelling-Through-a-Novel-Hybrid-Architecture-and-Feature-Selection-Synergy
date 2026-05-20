Welcome! This repository explains the implementation of the research work **"Advancing Tabular Stroke Modelling Through a Novel Hybrid Architecture and Feature‑Selection Synergy"**, published at the [2025 IEEE International Conference on Biomedical Engineering, Computer and Information Technology for Health](https://ieeexplore.ieee.org/abstract/document/11503962/) conference.

### [📖 Read The Full Paper](https://www.researchgate.net/publication/404670907_Advancing_Tabular_Stroke_Modelling_Through_a_Novel_Hybrid_Architecture_and_Feature-Selection_Synergy) &nbsp;|&nbsp; [📓 View the Notebook](main.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yousufislam191/Advancing-Tabular-Stroke-Modelling-Through-a-Novel-Hybrid-Architecture-and-Feature-Selection-Synergy/blob/main/main.ipynb)

---

## 1. What is this project about?

Brain stroke is a serious medical condition. Predicting stroke risk early can help doctors act quickly. This project uses **machine learning** to predict whether a patient is likely to have a stroke, based on simple health and lifestyle data.

We tackle two major challenges:

- **Imbalanced data** – Most people do not have strokes, so a model can become biased.
- **Feature selection** – Not all patient information is equally important.

The solution? We combine **advanced pre‑processing**, **smart feature selection**, and **ensemble models** (multiple ML models working together) to achieve very high prediction accuracy (ROC‑AUC up to 99.66%).

---

## 2. Dataset

- **Source**: [Kaggle – full‑filled brain stroke dataset](https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset)
- **Size**: 4,981 patient records
- **Features** (inputs):
    - Gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status.
- **Target** (output): `stroke` – 0 (no stroke) or 1 (stroke).
- **Class imbalance**: Only ~5% of records are stroke cases.

---

## 3. Implementation Workflow

Below we explain every major part of the Python code. The logic follows the paper exactly.

### 3.1. Environment & Libraries

We use Google Colab to run the code and the following main libraries:

- **pandas, numpy** – data handling
- **matplotlib, seaborn** – plots and visualisations
- **scikit‑learn** – ML models, metrics, preprocessing, cross‑validation
- **imblearn** – SMOTE (synthetic oversampling)
- **lightgbm, xgboost** – advanced tree‑based models
- **joblib, concurrent.futures** – parallel processing to save time

To run this project we recommend using **Google Colab**. Open the notebook using the badge above and run the dependency installation cell at the top of the notebook. In Colab, run this cell before executing the rest of the notebook:

```bash
# In a Colab notebook cell:
!pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn lightgbm xgboost joblib
```

If you prefer to run locally, create a virtual environment and run the same `pip install` command locally. (No `requirements.txt` is included because the workflow targets Colab.)

### 3.2. Data Loading and Exploratory Data Analysis (EDA)

- Load the CSV file with `pandas.read_csv()`.
- Print the shape (rows × columns), check for missing values, and display sample rows.
- Plot class distribution – a bar chart showing how many stroke vs non‑stroke patients we have.
- Visualise numerical features (`age`, `avg_glucose_level`, `bmi`) with histograms and note skewness.
- Visualise categorical features (`gender`, `ever_married`, etc.) with count plots, adding percentage labels.

**Why?** This helps us understand the data, spot outliers, and see if the classes are balanced.

### 3.3. Data Pre‑processing

#### 3.3.1. Encode Categorical Variables

Machine learning models understand numbers, not words. We convert text categories (like “Male”/“Female”) into numbers using **Label Encoding**. Each unique category gets an integer label (0, 1, 2, …). The mappings are stored so we can interpret them later.

#### 3.3.2. Remove Outliers (IQR Method)

Outliers are extreme values that can confuse a model. We use the **Interquartile Range (IQR)** rule:

- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 – Q1
- Lower bound = Q1 – 1.5 × IQR
- Upper bound = Q3 + 1.5 × IQR

Any value outside these bounds is removed. We do this only for numerical columns. In our data:

- **age**: 0 outliers
- **avg_glucose_level**: 602 outliers removed
- **bmi**: 42 outliers removed

After removal, the dataset has **4,337 rows**.

We also draw box‑plots before and after to see the difference.

#### 3.3.3. Feature Scaling (Standardisation)

Numerical features are standardised (z‑score normalisation). This means each numerical feature is transformed to have mean = 0 and standard deviation = 1. This helps distance‑based models (like KNN, SVM, neural networks) perform better.

#### 3.3.4. Handling Class Imbalance – SMOTE

Since only ~5% of patients had a stroke, a model could simply predict “no stroke” for everyone and still be 95% accurate! That’s useless. We fix this with **SMOTE (Synthetic Minority Oversampling Technique)**. It creates new synthetic stroke examples by interpolating between existing stroke cases. We apply SMOTE **only on the training data** to avoid leaking information into the test set.

After SMOTE, the number of stroke and non‑stroke samples in training are balanced (each class has 3,342 samples).

### 3.4. Feature Selection Strategies

We test three different feature sets to see which works best:

1. **Full Features** – Use all 10 features (after encoding, before splitting).
2. **Correlation‑based Selection** – Keep only features whose absolute correlation with `stroke` is ≥ 0.02. This gives 7 features: `age, hypertension, ever_married, heart_disease, work_type, bmi, smoking_status`.
3. **Random Forest‑based Selection** – Train a quick Random Forest, then keep features with importance ≥ 0.025. This selects 7 features: `avg_glucose_level, bmi, age, smoking_status, work_type, gender, Residence_type`.

For each strategy, we split the data into training (80%) and test (20%) sets, apply scaling and SMOTE only to training data.

### 3.5. Individual Classifiers

We train 10 different ML models on each feature set:

- Logistic Regression
- Random Forest
- Support Vector Classifier (SVC)
- Decision Tree
- K‑Nearest Neighbors (KNN)
- Gradient Boosting
- AdaBoost
- LightGBM
- XGBoost
- Multi‑layer Perceptron (Neural Network)

**Training**: A helper function trains each model in parallel (using `joblib`) to speed things up.

**Evaluation** on the test set: We compute accuracy, precision, recall, F1‑score, and ROC‑AUC (if the model can output probabilities). Results are printed in a table and also stored for later comparison.

**Cross‑validation**: To be sure the model performance is stable, we later run 5‑fold stratified cross‑validation and compute ROC‑AUC for each model. The results are sorted to see which models perform best.

**Model correlation**: We also check how similar the predictions of different models are (using correlation matrices). Low correlation means the models make different mistakes – a good sign for creating an ensemble later.

### 3.6. Ensemble Methods

Ensembles combine multiple base models to make a final, stronger prediction.

#### 3.6.1. Voting Classifiers

- **Hard Voting**: each model votes for a class; majority wins.
- **Soft Voting**: each model provides a probability for each class; the probabilities are averaged, and the class with the highest average probability is chosen.

We build both for every feature set, but soft voting usually performs better because it uses the confidence of each model.

#### 3.6.2. Stacking Ensemble

Stacking is more advanced:

- **Base models** (5–6 different classifiers) make predictions.
- A **meta‑model** (XGBoost) learns how to best combine those predictions.
- To prevent overfitting, the meta‑model is trained on out‑of‑fold predictions from cross‑validation.

We create a stacking ensemble for each feature set, using different base models suited to that set (tuned via grid search).

### 3.7. Hyperparameter Tuning

For the key models used in ensembles (Decision Tree, KNN, LightGBM, XGBoost, Random Forest, Gradient Boosting, Neural Network, Logistic Regression), we perform **GridSearchCV**. This automatically tries many combinations of parameters (like tree depth, learning rate) and picks the combination that gives the best ROC‑AUC using 5‑fold cross‑validation.

The best parameters are then used to build the final base models for the ensembles.

### 3.8. Model Benchmarking with Cross‑Validation

We evaluate all models – individual classifiers, soft voting, and stacking ensembles – using **5‑fold stratified cross‑validation** on the training data. This gives us:

- Mean ± standard deviation for Accuracy, F1‑score, and ROC‑AUC.
- A robust comparison of methods, showing that the Stacking Ensemble consistently achieves the highest scores and lowest variance.

Results are displayed as:

- Tables (mean ± std)
- Bar charts (comparing models per metric)
- Heatmaps (visualising the performance matrix)

### 3.9. Visualising Results

- **ROC curves** for all ensemble models on each feature set.
- **Box plots** showing the cross‑validation stability of top methods (Stacking Ensemble, Soft Voting, LightGBM) across folds. This is the `cv_stability.png` figure from the paper.
- **Model performance comparison** plots – bar charts and heatmaps.

---

## 4. Key Results Recap

| Method                       | ROC‑AUC (Mean ± Std) | Accuracy (Mean ± Std) | F1‑Score (Mean ± Std) |
| ---------------------------- | -------------------- | --------------------- | --------------------- |
| LightGBM (Full Features)     | 99.44% ± 0.51%       | 95.93% ± 1.87%        | 95.90% ± 2.04%        |
| Soft Voting Ensemble         | 99.41% ± 0.25%       | 95.99% ± 0.69%        | 96.09% ± 0.69%        |
| **Stacking Ensemble (Full)** | **99.66% ± 0.29%**   | **97.20% ± 1.57%**    | **97.15% ± 1.70%**    |

The Stacking Ensemble outperforms all other methods, proving that combining diverse models with a meta‑learner yields superior stroke prediction.

---

## 5. Usage

- **Run in Google Colab (recommended):** Click the "Open In Colab" badge at the top to open `main.ipynb` in Colab. If the notebook requires external data, upload the CSV file to the Colab session or mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
# then adjust file paths, e.g. '/content/drive/MyDrive/path/to/dataset.csv'
```

- **Run cells:** Execute cells top‑to‑bottom. The first code cell installs dependencies; run it before other cells.

- **Local run (optional):** If you need to run locally, create a virtual environment and install packages used in the notebook (same pip list as the Colab cell). Ensure your Python version is 3.9+.

## 6. Cite this work

If you use this code or results, please cite the paper and the DOI. A machine‑readable `CITATION.cff` is included. Suggested citation (BibTeX available in the `CITATION.cff`):

Y. Islam, "Advancing Tabular Stroke Modelling Through a Novel Hybrid Architecture and Feature‑Selection Synergy," 2025.

---

## 7. Contact

If you have questions, file an issue or open a pull request on this repository. You can also contact the author via their GitHub profile: `yousufislam191`.
