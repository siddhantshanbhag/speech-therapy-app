import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error

# Title of the Web App
st.title("Speech and Listening Therapy - Participant Data with Real-Time Predictions")

# Load train and test data
# Replace with your actual data files
train_data = pd.read_csv(r'C:\Users\c22100145\Dissertation\LA_train.csv')
test_data = pd.read_csv(r'C:\Users\c22100145\Dissertation\LA_test.csv')

test_data.drop(test_data.iloc[:, 8:58], axis=1, inplace=True)
test_data = test_data.rename(columns={'LI age 1': 'LA1', 'LI age 2': 'LA2', 'LI age 3': 'LA3', 'LI age 4': 'LA4'})


train_data.drop(train_data.iloc[:, 12:25], axis=1, inplace=True)
train_data = train_data.iloc[:-921]

train_data['LA6'] = train_data['LA6'].fillna(train_data['LA6'].mean())
train_data['LA7'] = train_data['LA7'].fillna(train_data['LA7'].mean())
train_data['LA8'] = train_data['LA8'].fillna(train_data['LA8'].mean())
train_data = train_data.drop('Unnamed: 25', axis=1)
train_data = train_data.astype({col: 'int' for col in train_data.select_dtypes(include='float').columns})




# Sidebar for participant selection
st.sidebar.header("Select a Participant")
selected_case = st.sidebar.selectbox("Participants", test_data["Case name"])

# Show participant details
selected_participant = test_data[test_data["Case name"] == selected_case]
st.write(f"### Participant Details for {selected_case}")
st.write(f"**Case Number:** {selected_participant['Case no.'].values[0]}")
st.write(f"**Current Age:** {selected_participant['Current age'].values[0]}")
st.write(f"**Gender:** {selected_participant['Gender'].values[0]}")

# Train-Test Data Preparation
X_train = train_data[['LA1', 'LA2', 'LA3', 'LA4']]  # Features
y_train = train_data[['LA5', 'LA6', 'LA7', 'LA8']]  # Targets

X_test = selected_participant[['LA1', 'LA2', 'LA3', 'LA4']]

# Train models
def train_models(X_train, y_train):
    rf_model = RandomForestRegressor()
    ridge_model = Ridge()
    lasso_model = Lasso()

    # Train the models on X_train and y_train for each Listening Age (LA5, LA6, LA7, LA8)
    predictions = {}

    for i in range(y_train.shape[1]):  # Iterate over LI age 5, 6, 7, 8
        # Random Forest
        rf_model.fit(X_train, y_train.iloc[:, i])
        predictions[f'rf_LI_{i+5}'] = rf_model.predict(X_test)

        # Ridge Regression
        ridge_model.fit(X_train, y_train.iloc[:, i])
        predictions[f'ridge_LI_{i+5}'] = ridge_model.predict(X_test)

        # Lasso Regression
        lasso_model.fit(X_train, y_train.iloc[:, i])
        predictions[f'lasso_LI_{i+5}'] = lasso_model.predict(X_test)

    return predictions

# Make predictions
predictions = train_models(X_train, y_train)

# Display predictions
st.write("### Predictions for Listening Ages 5-8")
for model_name, pred in predictions.items():
    st.write(f"{model_name}: {pred[0]}")  # Display predictions

# Combine actual and predicted LI ages
li_actual = selected_participant[['LA1', 'LA2', 'LA3', 'LA4']].values.flatten()
li_predicted_rf = [predictions[f'rf_LI_{i}'][0] for i in range(5, 9)]
li_predicted_ridge = [predictions[f'ridge_LI_{i}'][0] for i in range(5, 9)]
li_predicted_lasso = [predictions[f'lasso_LI_{i}'][0] for i in range(5, 9)]

# Plot Listening Ages (LA1-LA8)
st.write("### Listening Age Progression (Random Forest Prediction)")

fig, ax = plt.subplots()
x = range(1, 9)
ax.plot(x[:4], li_actual, marker='o', label='Actual LI ages 1-4', color='blue')
ax.plot(x[4:], li_predicted_ridge, marker='x', label='Predicted LI ages 5-8 (Random Forest)', color='red')
ax.set_title(f"Listening Ages for {selected_case}")
ax.set_xlabel("Listening Age (LA)")
ax.set_ylabel("Value")
ax.legend()

# Display the graph
st.pyplot(fig)

# Optionally, add more graphs for Ridge and Lasso predictions
