import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Set the aesthetics for plots
sns.set(style="whitegrid")

# Set a custom CSS style for background
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f8ff;  /* Light blue background */
    }
    .sidebar .sidebar-content {
        background: #f0f8ff;  /* Ensure sidebar has the same background */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the Web App
st.title("Speech and Listening Therapy - Participant Data with Real-Time Predictions")

# Center the dropdown for selecting case numbers
st.markdown("<h2 style='text-align: center;'>Select a Participant Case Number</h2>", unsafe_allow_html=True)

# Load train and test data
# Replace with your actual data files
train_data = pd.read_csv(r'C:\Users\c22100145\Dissertation\LA_train.csv')
test_data = pd.read_csv(r'C:\Users\c22100145\Dissertation\LA_test.csv')

# Data Preprocessing
test_data.drop(test_data.iloc[:, 8:58], axis=1, inplace=True)

# Convert Case Number to String for Consistency
test_data['Case no.'] = test_data['Case no.'].astype(str)

train_data.drop(train_data.iloc[:, 12:25], axis=1, inplace=True)

# Populate case numbers for the dropdown
case_numbers = [""] + test_data["Case no."].tolist()  # Add an empty option for no selection
selected_case_number = st.selectbox("", case_numbers)  # Use Case Number for selection

# Initialize variables
selected_participant = None
predictions = {}

if selected_case_number:
    # Show participant details directly in a table
    selected_participant = test_data[test_data["Case no."] == selected_case_number]

    if not selected_participant.empty:
        # Create a DataFrame for participant details
        participant_details = {
            "Detail": ["Case Number", "Case Name", "Current Age", "Gender"],
            "Information": [
                selected_participant['Case no.'].values[0],
                selected_participant['Case name'].values[0],  # Add Case Name
                f"{selected_participant['Current age'].values[0]} years",
                selected_participant['Gender'].values[0]
            ]
        }
        
        details_df = pd.DataFrame(participant_details)
        
        st.write("### Participant Details")
        st.table(details_df)  # Display participant details in a table

        # Display current learning ages in a table
        current_learning_ages = selected_participant[['LA1', 'LA2', 'LA3', 'LA4']].values.flatten()
        current_learning_ages_labels = ['LA1', 'LA2', 'LA3', 'LA4']
        
        current_ages_df = pd.DataFrame({
            'Listening Age': current_learning_ages_labels,
            'Age (months)': current_learning_ages
        })

        st.write("### Current Learning Ages")
        st.table(current_ages_df)  # Display current ages in a table

        # Plot current learning ages with improved aesthetics
        st.write("### Current Learning Ages Plot")
        
        plt.figure(figsize=(8, 4))
        sns.barplot(x=current_learning_ages_labels, y=current_learning_ages, palette="Blues_d")
        plt.title(f"Current Learning Ages for {selected_participant['Case name'].values[0]}", fontsize=16)
        plt.ylabel("Age (months)", fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Line plot for current learning ages
        st.write("### Line Plot of Current Learning Ages")
        plt.figure(figsize=(8, 4))
        plt.plot(current_learning_ages_labels, current_learning_ages, marker='o', linestyle='-', color='blue', label='Current Ages')
        plt.title("Current Learning Ages - Line Plot", fontsize=16)
        plt.ylabel("Age (months)", fontsize=12)
        plt.xlabel("Listening Ages", fontsize=12)
        plt.axhline(y=current_learning_ages.mean(), color='orange', linestyle='--', label='Mean Age')
        plt.legend()
        st.pyplot(plt)

        # Show Forecast button only if a participant is selected
        if st.button("Forecast"):
            # Train-Test Data Preparation
            X_train = train_data[['LA1', 'LA2', 'LA3', 'LA4']]  # Features
            y_train = train_data[['LA5', 'LA6', 'LA7', 'LA8']]  # Targets

            # Prepare test data for prediction
            X_test = selected_participant[['LA1', 'LA2', 'LA3', 'LA4']]

            # Train Random Forest model
            rf_model = RandomForestRegressor()
            rf_model.fit(X_train, y_train)

            # Make predictions
            predictions = rf_model.predict(X_test)

            # Ensure predictions are in the correct format for display
            predictions = predictions.flatten()  # Flatten the predictions to 1D array

            # Display predictions in a similar table format to current ages
            predicted_labels = [f"LA{i+5}" for i in range(len(predictions))]
            predicted_ages_df = pd.DataFrame({
                'Listening Age': predicted_labels,
                'Predicted Age (months)': [int(pred) for pred in predictions]
            })

            st.write("### Predicted Learning Ages")
            st.table(predicted_ages_df)  # Display predicted ages in a table

            # Plot Learning Ages Progression
            st.write("### Learning Ages Progression Plot")
            
            # Combine current and predicted learning ages
            all_learning_ages = np.concatenate([current_learning_ages[:3], [current_learning_ages[3]], predictions])
            all_learning_ages_labels = current_learning_ages_labels + [f"LA5 ({int(predictions[0])} months)", f"LA6 ({int(predictions[1])} months)", f"LA7 ({int(predictions[2])} months)", f"LA8 ({int(predictions[3])} months)"]

            plt.figure(figsize=(10, 5))
            plt.plot(range(1, 5), current_learning_ages[:4], marker='o', color='blue', label='Current Ages', linewidth=2)
            plt.plot(range(5, 9), predictions, marker='x', color='orange', label='Predicted Ages', linewidth=2)
            plt.xticks(range(1, len(all_learning_ages)+1), all_learning_ages_labels, rotation=45)
            plt.title(f"Learning Ages Progression for {selected_participant['Case name'].values[0]}", fontsize=16)
            plt.ylabel("Age (months)", fontsize=12)
            plt.xlabel("Listening Ages", fontsize=12)
            plt.axvline(x=4.5, color='red', linestyle='--', label='Transition from Current to Predicted Ages')  # Add a line for better visualization
            plt.legend()
            st.pyplot(plt)
    else:
        st.warning("No participant found with the selected case number.")
