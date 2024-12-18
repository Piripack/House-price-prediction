<div class="section">
    <h2>Required Libraries</h2>
    <div class="code-container">
        <code>pip install scikit-learn</code><br>
        <code>pip install pandas</code><br>
        <code>pip install numpy</code><br>
        <code>pip install matplotlib</code><br>
        <code>pip install seaborn</code>
    </div>
</div>

<div class="section">
    <h2>Key Code Snippets</h2>

    <h3>1. Data Preprocessing</h3>
    <div class="code-container" id="code1">
        <code>
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Average-prices-data.csv")

# Handle missing data
df = df.dropna()

# Feature selection
features = df[['PropertyType', 'Location', 'Year', 'Size']]
target = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardizing data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
        </code>
    </div>
    <button class="copy-button" onclick="copyToClipboard('code1')">Copy to Clipboard</button>

    <h3>2. Model Building and Evaluation</h3>
    <div class="code-container" id="code2">
        <code>
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")
        </code>
    </div>
    <button class="copy-button" onclick="copyToClipboard('code2')">Copy to Clipboard</button>
</div>

<div class="section">
    <h2>Conclusion</h2>
    <p>This project demonstrates the power of predictive modeling using the Random Forest Regressor for house price prediction. Its applications extend to industries like finance and cybersecurity, offering significant potential for data-driven decision-making. Follow this guide to replicate the analysis and adapt the model to different datasets.</p>
</div>

<div class="download-section">
    <a href="https://example.com/dataset.csv" class="download-button">
        <i class="fas fa-download"></i> Download Dataset
    </a>
    <div class="download-info">
        UK HPI full file (CSV, 60MB) From:<br>
        HM Land Registry<br>
        Published 14 February 2024
    </div>
</div>
