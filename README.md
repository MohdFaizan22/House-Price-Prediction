# 🏠 Bengaluru House Price Prediction

An end-to-end Machine Learning project that predicts house prices in Bengaluru using features like square footage, BHK, bathrooms, and locality-based pricing.

Built using:
- Python
- Pandas
- Scikit-learn
- XGBoost
- Flask
- Streamlit

---

# 📌 Project Overview

This project uses the Bengaluru House Price dataset to build a real estate price prediction system.

The workflow includes:
- Data cleaning
- Feature engineering
- Outlier removal
- Log transformation
- Model training
- Hyperparameter tuning
- Feature importance analysis
- Flask API development
- Streamlit frontend deployment

---

# 🚀 Features

✅ Cleaned messy real-estate data  
✅ Converted sqft ranges into numeric values  
✅ Engineered geospatial locality pricing feature  
✅ Removed outliers using IQR method  
✅ Applied log transformation to target variable  
✅ Compared multiple ML models  
✅ Tuned XGBoost hyperparameters  
✅ Built Flask prediction API  
✅ Created interactive frontend UI  
✅ Ready for deployment on Hugging Face / Render  

---

# 📂 Dataset

Dataset used:
- Bengaluru House Price Dataset

Main features:
- location
- total_sqft
- bath
- balcony
- BHK
- price

---

# 🧹 Data Preprocessing

## 1. Handling Missing Values

Removed null values and inconsistent rows.

```python
data = data.dropna()
```

---

## 2. Converted `size` Column to BHK

Example:

```python
2 BHK → 2
```

Code:

```python
data['bhk'] = data['size'].apply(
    lambda x: int(str(x).split()[0])
)
```

---

## 3. Cleaned `total_sqft`

Handled:
- ranges
- inconsistent units
- invalid values

Examples:

```python
2100 - 2850 → 2475
```

Code:

```python
def convert_sqft(x):

    x = str(x)

    if '-' in x:
        a, b = x.split('-')
        return (float(a) + float(b)) / 2

    try:
        return float(x)

    except:
        return None
```

Applied:

```python
data['total_sqft'] = data['total_sqft'].apply(convert_sqft)
```

---

# ⚙️ Feature Engineering

## 1. Price Per Sqft

Created normalized pricing feature:

```python
data['price_per_sqft'] = (
    data['price'] * 100000
) / data['total_sqft']
```

Used for:
- outlier detection
- normalization
- better model learning

---

## 2. Geospatial Locality Feature

Calculated average locality price using:

```python
location_price = data.groupby(
    'location'
)['price'].mean()
```

Mapped back to dataset:

```python
data['location_avg_price'] = data[
    'location'
].map(location_price)
```

This feature helps the model learn:
- expensive locations
- cheaper localities
- pricing trends by area

---

# 📊 Outlier Removal

Used IQR (Interquartile Range) method.

Formula:

```python
IQR = Q3 - Q1
```

Outlier Range:

```python
[Q1 - 1.5(IQR), Q3 + 1.5(IQR)]
```

Code:

```python
Q1 = data['price_per_sqft'].quantile(0.25)

Q3 = data['price_per_sqft'].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

data = data[
    (data['price_per_sqft'] >= lower_limit) &
    (data['price_per_sqft'] <= upper_limit)
]
```

---

# 📈 Log Transformation

Applied logarithmic transformation on target variable:

```python
import numpy as np

data['log_price'] = np.log(data['price'])
```

Benefits:
- reduced skewness
- stabilized variance
- improved regression performance

---

# 🤖 Machine Learning Models

Compared:
- Linear Regression
- Ridge Regression
- XGBoost Regressor

---

# 📌 Feature & Target Selection

```python
X = data[
    [
        'total_sqft',
        'bath',
        'bhk',
        'location_avg_price'
    ]
]

y = data['log_price']
```

---

# ✂️ Train Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 📊 Cross Validation Evaluation

Used:

```python
cross_val_score()
```

Scoring Metric:
- R² Score

---

# 📈 Model Results

| Model | R² Score |
|---|---|
| Linear Regression | 0.559 |
| Ridge Regression | 0.559 |
| XGBoost | 0.827 |

---

# 🏆 Best Model

## XGBoost Regressor

Reason:
- captures non-linear relationships
- handles feature interactions
- performs well on tabular datasets

---

# 🔧 Hyperparameter Tuning

Used:

```python
GridSearchCV
```

Parameter Grid:

```python
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
```

Best Parameters:

```python
{
 'learning_rate': 0.1,
 'max_depth': 7,
 'n_estimators': 100
}
```

Best Tuned Score:

```python
0.823
```

---

# 📌 Feature Importance

Visualized feature importance using XGBoost.

Top contributing features:
- Location Average Price
- Total Square Feet
- BHK
- Bathrooms

Code:

```python
import matplotlib.pyplot as plt

importance = xgb.feature_importances_

features = X.columns

plt.figure(figsize=(8,5))

plt.bar(features, importance)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.title("Feature Importance")

plt.show()
```

---

# 🌐 Flask API

Created a Flask API for predictions.

## POST Endpoint

```python
/predict
```

---

## Example Request

```json
{
  "location": 85,
  "BHK": 2,
  "area": 1200,
  "bath": 2
}
```

---

## Example Response

```json
{
  "predicted_price": 78.5
}
```

---

# 🖥️ Streamlit Frontend

Built an interactive UI using Streamlit.

Features:
- Area input
- Bathroom input
- BHK input
- Location pricing input
- Instant prediction display

---

# 📦 Installation

Clone repository:

```bash
git clone https://github.com/your-username/house-price-predictor.git
```

Move into project directory:

```bash
cd house-price-predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Flask App

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# ▶️ Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

# 📁 Project Structure

```text
house-price-predictor/
│
├── app.py
├── streamlit_app.py
├── house_price_model.pkl
├── requirements.txt
├── runtime.txt
├── README.md
```

---

# 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Matplotlib | Visualization |
| Scikit-learn | ML utilities |
| XGBoost | Regression model |
| Flask | API backend |
| Streamlit | Frontend UI |

---

# 📚 Key Learnings

- Real-world data preprocessing
- Feature engineering
- Outlier handling using IQR
- Log transformation
- Model comparison using cross-validation
- Hyperparameter tuning
- Flask API creation
- Streamlit UI development
- ML deployment workflow

---

# 🔮 Future Improvements

- Use actual location names
- Add location dropdown
- Add map-based visualization
- Improve frontend UI
- Add cloud deployment pipeline
- Add model monitoring

---

# 👨‍💻 Author

Mohd Faizanullah

Aspiring ML Engineer focused on:
- Machine Learning
- Deep Learning
- AI Applications
- Full ML Deployment Pipelines

