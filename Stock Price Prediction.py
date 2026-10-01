import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("d:\Lakshay\Datasets_csv\stock_data_readable.csv")



# ==========================================
# SEPARATE FEATURES AND TARGET
# ==========================================

x = df.drop(columns="Price_Direction")
y = df["Price_Direction"]


# ==========================================
# SPLIT DATA
# ==========================================

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# FEATURE SCALING
# ==========================================

scale = StandardScaler()

x_train_scale = scale.fit_transform(x_train)
x_test_scale = scale.transform(x_test)


# ==========================================
# HYPERPARAMETER GRID
# ==========================================

para_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

tuning_model = RandomForestClassifier(
    random_state=42
)


# ==========================================
# GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    param_grid=para_grid,
    estimator=tuning_model,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1
)


print("\n==========================================")
print("Please wait while the model is getting ready...")
print("GridSearchCV is training the model.")
print("This may take several minutes.")
print("==========================================\n")


# ==========================================
# TRAIN MODEL
# ==========================================

grid_search.fit(x_train_scale, y_train)


print("\n==========================================")
print("Model is ready!")
print("==========================================\n")


# ==========================================
# BEST PARAMETERS
# ==========================================

print("Best Parameters:")
print(grid_search.best_params_)


# ==========================================
# BEST MODEL
# ==========================================

model = grid_search.best_estimator_


# ==========================================
# TEST PREDICTION
# ==========================================

y_pred = model.predict(x_test_scale)


# ==========================================
# TEST ACCURACY
# ==========================================

test_accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", test_accuracy)


# ==========================================
# TRAINING ACCURACY
# ==========================================

y_train_pred = model.predict(x_train_scale)

train_accuracy = accuracy_score(y_train, y_train_pred)

print("Training Accuracy:", train_accuracy)


# ==========================================
# USER INPUT FOR PREDICTION
# ==========================================

print("\n==========================================")
print("Enter Stock Information")
print("==========================================\n")


Opening_Price = float(
    input("Enter the opening price of stock: ")
)

Closing_Price = float(
    input("Enter the closing price of stock: ")
)

Highest_Price = float(
    input("Enter the highest price of stock: ")
)

Lowest_Price = float(
    input("Enter the lowest price of stock: ")
)

Trading_Volume = float(
    input("Enter the trading volume of stock: ")
)

Relative_Strength_Index = float(
    input("Enter the relative strength index: ")
)

Moving_Average_Convergence_Divergence = float(
    input("Enter the moving average convergence divergence: ")
)

Upper_Bollinger_Band = float(
    input("Enter the upper Bollinger band: ")
)

Lower_Bollinger_Band = float(
    input("Enter the lower Bollinger band: ")
)

Market_Sentiment_Score = float(
    input("Enter the market sentiment score: ")
)

GDP_Growth_Rate = float(
    input("Enter the GDP growth rate: ")
)

Inflation_Rate = float(
    input("Enter the inflation rate: ")
)


# ==========================================
# CREATE DATAFRAME FOR NEW INPUT
# ==========================================

new_data = pd.DataFrame([{
    "Opening_Price": Opening_Price,
    "Closing_Price": Closing_Price,
    "Highest_Price": Highest_Price,
    "Lowest_Price": Lowest_Price,
    "Trading_Volume": Trading_Volume,
    "Relative_Strength_Index": Relative_Strength_Index,
    "Moving_Average_Convergence_Divergence":
        Moving_Average_Convergence_Divergence,
    "Upper_Bollinger_Band": Upper_Bollinger_Band,
    "Lower_Bollinger_Band": Lower_Bollinger_Band,
    "Market_Sentiment_Score": Market_Sentiment_Score,
    "GDP_Growth_Rate": GDP_Growth_Rate,
    "Inflation_Rate": Inflation_Rate
}])


# ==========================================
# SCALE NEW INPUT
# ==========================================

new_data_scaled = scale.transform(new_data)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(new_data_scaled)


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n==========================================")

if prediction[0] == 0:
    print("The stock price will not increase")
else:
    print("The stock price will increase")

print("==========================================")