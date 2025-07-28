# Car Price Prediction using Machine Learning

This project explores multiple machine learning regression models to predict the price of used cars based on their specifications. The dataset is processed and cleaned, followed by feature engineering, encoding, and model tuning using `GridSearchCV`.

## Objective

To accurately predict the price of a car given its specifications such as drivetrain, power, torque, transmission type, fuel type, and more.

## Dataset

- **Source**: `car_details.csv`
- **Target variable**: `Price`
- **Features used**:  
  - Year, Kilometers Driven, Drivetrain, Owner, Transmission, Seller Type, Make, Fuel Type, Car Dimensions, Seating Capacity, Fuel Tank Capacity, Power, Torque, Engine Capacity, etc.

Custom feature extraction functions were written to handle text columns such as Max Power, Torque, and Engine into numerical formats (e.g., extracting BHP and RPM).

## Preprocessing Steps

- Extraction of numeric features from text (`Max Power`, `Torque`, `Engine`)
- Handling rare categories (Make column grouped under "Other" if low frequency)
- Missing value treatment (dropped rows with missing values)
- Encoding:
  - Ordinal Encoding for `Drivetrain` and `Owner`
  - One-Hot Encoding for nominal categorical features
- Feature scaling using `StandardScaler`

## Models Trained

Four regression models were trained using `Pipeline` and `GridSearchCV` for hyperparameter tuning:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Support Vector Regressor (SVR)**
4. **Gradient Boosting Regressor**

## Evaluation Metric

- **R² Score** on the test set

##  Model Performance (on Test Set)

| Model                      | R² Score |
|---------------------------|----------|
| Linear Regression         | 0.6141   |
| Random Forest Regressor   | 0.9241   |
| Support Vector Regressor  | -241153333.5948 *(failed)* |
| Gradient Boosting Regressor | 0.9491 |

> Note: SVR model failed to converge or was not suitable for the scale/distribution of this dataset.
