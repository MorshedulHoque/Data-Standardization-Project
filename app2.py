import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_data(df):
    # Separate features and target variable
    # X = df.drop('target_column', axis=1)  # Adjust 'target_column' to your target variable

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to the features and transform the data
    X_standardized = scaler.fit_transform(df)

    # Create a new DataFrame with standardized data
    df_standardized = pd.DataFrame(X_standardized, columns=df.columns)

    # Add the target variable back to the DataFrame
    # df_standardized['target_column'] = df['target_column']

    return df_standardized

def main():
    st.title("CSV Data Standardization App")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Display the original dataset
        st.subheader("Original Dataset")
        st.write(df)

        # Standardize the data
        df_standardized = standardize_data(df)

        # Display the standardized dataset
        st.subheader("Standardized Dataset")
        st.write(df_standardized)

if __name__ == "__main__":
    main()
