import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_data(input_file, output_file):
    # Load the dataset
    df = pd.read_csv(input_file)

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

    # Save the standardized data to a new CSV file
    df_standardized.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "D:\Data Science Projects\Demo project\Custom.csv"
    output_file = "D:\Data Science Projects\Demo project\standardized_dataset.csv"

    standardize_data(input_file, output_file)
