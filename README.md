# Stock Price Prediction using Machine Learning

## About the Project

This project uses Machine Learning to predict stock price direction using Python and Scikit-learn. It uses a Random Forest Classifier to predict whether the stock price will increase or not.

The project includes data preprocessing, feature scaling, hyperparameter tuning, and model evaluation.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Classifier
* GridSearchCV
* StandardScaler

## Dataset

The dataset contains stock market features such as:

* Opening Price
* Closing Price
* Highest Price
* Lowest Price
* Trading Volume
* Relative Strength Index (RSI)
* Moving Average Convergence Divergence (MACD)
* Upper and Lower Bollinger Bands
* Market Sentiment Score
* GDP Growth Rate
* Inflation Rate

**Target Column:** `Price_Direction`

## Project Workflow

1. Load and explore the dataset using Pandas.
2. Separate features and target variable.
3. Split the dataset into training and testing sets.
4. Scale features using StandardScaler.
5. Train a Random Forest Classifier.
6. Use GridSearchCV for hyperparameter tuning.
7. Evaluate the model using accuracy score.
8. Predict stock price direction using user input.

## Model Performance

* Algorithm: Random Forest Classifier
* Hyperparameter Tuning: GridSearchCV
* Test Accuracy: 95.75%

## How to Run

**1. Clone the repository**

```bash
git clone https://github.com/wadhwalakshay08-create/stock-price-prediction.git
```

**2. Navigate to the project folder**

```bash
cd stock-price-prediction
```

**3. Install the required libraries**

```bash
pip install pandas numpy scikit-learn
```

**4. Run the Python file**

```bash
python "Stock Price Prediction.py"
```

Make sure `stock_data_readable.csv` is in the same folder as the Python file.

## Features

* Stock price direction prediction
* Data preprocessing and feature scaling
* Random Forest classification
* Hyperparameter tuning using GridSearchCV
* Model evaluation
* User input-based prediction

## Note

This is an educational Machine Learning project. Its predictions are not financial advice and should not be used alone for investment decisions.

## Author

Lakshay Wadhwa
