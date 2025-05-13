# Car Price Prediction Using Machine Learning

This project aims to predict car prices using machine learning models. The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data) and contains various features about cars, such as engine power, torque, dimensions, drivetrain type, year of manufacture, kilometers driven, and others. The goal is to predict the car price based on these features using two different machine learning algorithms: Linear Regression and Random Forest Regression.

## Project Overview

The project utilizes the following steps:

1. **Data Preprocessing**:

   * The dataset is cleaned by extracting relevant features from string-based columns (e.g., max power, torque, engine size, and drivetrain type).
   * Missing values are removed from the dataset.
   * New columns are derived to represent numeric values extracted from the raw data.

2. **Feature Engineering**:

   * Features like `Max Power`, `Max Torque`, `Engine Capacity`, and `Drivetrain Type` are extracted into separate numeric columns.
   * The relevant features are selected for regression models, while the target variable (`Price`) is separated.

3. **Model Training**:

   * **Linear Regression**: A simple linear regression model is trained to predict car prices based on the features.
   * **Random Forest Regression**: A Random Forest model, an ensemble method, is trained to predict the same target variable.

4. **Evaluation**:

   * Both models' performance is evaluated using the **R-squared (R²)** metric, which indicates the proportion of variance explained by the model.
   * The model with the highest R² score is considered to have better predictive power.

## Data Preprocessing

The dataset is loaded from a CSV file `car_details.csv`, and the following transformations are applied:

* **Power and Torque Extraction**: The `Max Power` and `Max Torque` columns contain values like "150 bhp @ 6000 rpm" and "200 Nm @ 3500 rpm". Regular expressions are used to extract these values into separate columns: `power_bhp`, `power_rpm`, `torque_nm`, and `torque_rpm`.

* **Engine Capacity**: The `Engine` column contains information like "1500 cc". This value is extracted into a new column `engine_power`.

* **Drivetrain Type**: The `Drivetrain` column (e.g., "FWD", "RWD") is converted into numeric values (2 for FWD, 3 for RWD, and 4 for other).

## Models and Results

* **Linear Regression**:

  * A simple linear regression model was trained using the selected features.
  * The R² score for the Linear Regression model was calculated to assess its performance.

* **Random Forest Regression**:

  * A Random Forest Regressor model was trained to capture non-linear relationships between the features and the target variable.
  * The R² score for the Random Forest model was calculated.

### Model Comparison

* **Linear Regression R² Score**: `lr_r2`
* **Random Forest R² Score**: `rf_r2`

The **Random Forest** model outperforms **Linear Regression** in predicting car prices, as it captures more complex relationships between the features. This can be attributed to the ability of Random Forest to handle non-linearities and interactions between multiple features, unlike Linear Regression, which assumes a linear relationship between the features and the target.

## Conclusion

The Random Forest model provides better predictive accuracy compared to Linear Regression, indicating that more complex models like Random Forest are more suitable for this task. Future work could involve fine-tuning the Random Forest model or exploring other machine learning models to improve predictions further.

## Dataset

The dataset used in this project was sourced from Kaggle. You can find the dataset [here](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho/data). (I have used the car details v4.csv file)

## Requirements

To run this project, the following Python libraries are required:

* `pandas`
* `numpy`
* `scikit-learn`
* `re` (for regular expressions)

You can install the required libraries using:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Clone the repository or download the project files.
2. Place the `car_details.csv` file in the appropriate directory.
3. Run the Python script to see the model results.
