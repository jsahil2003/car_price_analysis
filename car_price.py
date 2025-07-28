import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn import set_config
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
set_config(display='diagram')

# Load dataset
df = pd.read_csv('car_details.csv')

# Function to extract numerical power values
def extract_power(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*bhp\s*@\s*(\d+)', value)
        if match:
            bhp = float(match.group(1))
            power_rpm = int(match.group(2))
            return pd.Series([bhp, power_rpm])
    return pd.Series([None, None])

# Function to extract torque
def extract_torque(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*Nm\s*@\s*(\d+)', value)
        if match:
            nm = float(match.group(1))
            rpm = int(match.group(2))
            return pd.Series([nm, rpm])
    return pd.Series([None, None])

# Function to extract engine power
def extract_engine(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*cc', value)
        if match:
            cc = float(match.group(1))
            return pd.Series([cc])
    return pd.Series([None])

# Group rare car makes under 'Other'
value_counts = df['Make'].value_counts()
threshold = 25
common_make = value_counts[value_counts > threshold].index
df['Make'] = df['Make'].apply(lambda x: x if x in common_make else 'Other')

# Extract numerical values from text columns
df[['power_bhp', 'power_rpm']] = df['Max Power'].apply(extract_power)
df[['torque_nm', 'torque_rpm']] = df['Max Torque'].apply(extract_torque)
df['engine_power'] = df['Engine'].apply(extract_engine)
df.drop(columns=['Max Torque', 'Max Power', 'Engine'], inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

# Train-test split
x = df.drop(columns=['Price', 'Model', 'Location', 'Color'])
y = df['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# ColumnTransformer pipeline
preprocessor = ColumnTransformer(transformers=[
    ('ord1', OrdinalEncoder(categories=[['FWD', 'RWD', 'AWD']], handle_unknown='use_encoded_value', unknown_value=3), ['Drivetrain']),
    ('ord2', OrdinalEncoder(categories=[['Third', 'Second', 'First', 'UnRegistered Car']], handle_unknown='use_encoded_value', unknown_value=4), ['Owner']),
    ('nom', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='infrequent_if_exist'), ['Transmission', 'Seller Type', 'Make', 'Fuel Type']),
    ('ss', StandardScaler(), ['Year', 'Kilometer', 'Length', 'Width', 'Height',
                              'Seating Capacity', 'Fuel Tank Capacity', 'power_bhp', 'power_rpm',
                              'torque_nm', 'torque_rpm', 'engine_power'])
], remainder='drop')

# Model dictionary with parameters
models = {
    'LinearRegression': {
        'model': LinearRegression(),
        'param': {}
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(),
        'param': {
            'clf__n_estimators': [150, 200, 250],
            'clf__criterion': ['squared_error'],
            'clf__max_depth': [15, 21, 27]
        }
    },
    'SVR': {
        'model': SVR(),
        'param': {
            'clf__C': [0.3, 0.5, 0.7, 0.9]
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(),
        'param': {
            'clf__loss': ['squared_error'],
            'clf__n_estimators': [150, 200, 250],
            'clf__max_depth': [None, 10, 15, 20],
            'clf__learning_rate': [0.1, 0.3, 0.6]
        }
    }
}

# Dictionary to store grid search results
grid_scores = {}

# Fit each model with GridSearchCV
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('Preprocessing', preprocessor),
        ('clf', model['model'])
    ])
    grid = GridSearchCV(pipe, param_grid=model['param'], cv=5, n_jobs=-1, verbose=1)
    grid.fit(x_train, y_train)

    grid_scores[name] = {
        'best_score': grid.best_score_,
        'best_model': grid.best_estimator_,
        'best_params': grid.best_params_,
        'grid_object': grid
    }
    print(f"{name} - Best CV Score = {grid.best_score_}")

# Print best parameters and CV scores
for name, result in grid_scores.items():
    print(f"\n{name}:")
    print(f"Score : {result['best_score']:.4f}")
    print("Parameters:")
    for param, value in result['best_params'].items():
        print(f"{param} : {value}")

# Test performance on x_test
for name, result in grid_scores.items():
    best_model = result['grid_object'].best_estimator_
    y_pred = best_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"Test Accuracy : {r2}")
