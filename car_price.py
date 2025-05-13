# import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
#import xgboost as xgb
import re

# read data
df = pd.read_csv('/Users/sahiljadhav/Documents/ML using ChatGPT/car_details.csv')



# create functions to extract required values which are grouped together with strings
def extract_power(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*bhp\s*@\s*(\d+)',value)
        if match:
            bhp = float(match.group(1))
            power_rpm = int(match.group(2))
            return pd.Series([bhp,power_rpm]) 
    return pd.Series([None,None])

def extract_torque(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*Nm\s*@\s*(\d+)',value)
        if match:
            nm = float(match.group(1))
            rpm = int(match.group(2))
            return pd.Series([nm,rpm])
    return pd.Series([None,None])

def extract_engine(value):
    if isinstance(value, str):
        match = re.search(r'([\d.]+)\s*cc' , value)
        if match:
            cc = float(match.group(1))
            return pd.Series([cc])
    return pd.Series([None])

def drive(value):
    if isinstance(value, str):
        if value == 'FWD':
            return pd.Series([2])
        elif value == 'RWD':
            return pd.Series([3])
        else:
            return pd.Series([4])
    return pd.Series([None])



# add columns to use them for regression
df[['power_bhp','power_rpm']] = df['Max Power'].apply(extract_power)
df[['torque_nm' , 'torque_rpm']] = df['Max Torque'].apply(extract_torque)
df['engine_power'] = df['Engine'].apply(extract_engine)
df['drive_type'] = df['Drivetrain'].apply(drive)
df = df.drop(columns = ['Max Torque' , 'Max Power', 'Engine' , 'Drivetrain' ])

# defining a cleaned df
df_clean = df[['power_bhp' , 'power_rpm' , 'torque_nm' , 'torque_rpm' ,
              'engine_power' , 'drive_type' , 'Year' , 'Kilometer' ,
              'Length' , 'Width' , 'Height' , 'Seating Capacity' , 'Fuel Tank Capacity' , 'Price']].dropna()
factors = df_clean.drop(columns = 'Price')
price = df_clean['Price']

# spliting the data into training and testing
x_train , x_test , y_train , y_test = train_test_split(factors, price, test_size = 0.2, random_state = 42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train,y_train)
lr_predictions = lr_model.predict(x_test)  
lr_r2 = r2_score(y_test,lr_predictions)
lr_r2

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(x_train,y_train)
rf_predictions = rf_model.predict(x_test)
rf_r2 = r2_score(y_test,rf_predictions)
rf_r2

# Random Forest captures the relationship better