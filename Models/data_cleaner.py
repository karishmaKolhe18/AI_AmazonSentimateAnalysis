import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def clean_data(df, missing_value_strategy="mean", scale_method=None):
    """
    Cleans a given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        missing_value_strategy (str): "mean", "median", "mode", or "drop" to handle missing values.
        scale_method (str): "standard" (StandardScaler) or "minmax" (MinMaxScaler).

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    # 1. Remove duplicates
    df = df.drop_duplicates()

    # 2. Handle missing values
    if missing_value_strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif missing_value_strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif missing_value_strategy == "mode":
        df = df.fillna(df.mode().iloc[0])
    elif missing_value_strategy == "drop":
        df = df.dropna()

    # 3. Convert data types
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass  # Keep as object if conversion fails

    # 4. Remove outliers using IQR method
    Q1 = df.quantile(0.25, numeric_only=True)
    Q3 = df.quantile(0.75, numeric_only=True)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 5. Normalize or standardize data
    if scale_method:
        scaler = StandardScaler() if scale_method == "standard" else MinMaxScaler()
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df

if __name__ == "__main__":
    # Load data from CSV file
    input_file = "data.csv"  # Change this to your file name
    output_file = "cleaned_data.csv"

    print("Loading dataset...")
    df = pd.read_csv(input_file)

    print("Cleaning data...")
    cleaned_df = clean_data(df, missing_value_strategy="median", scale_method="minmax")

    print("Saving cleaned data...")
    cleaned_df.to_csv(output_file, index=False)

    print(f"Data cleaning complete! Cleaned data saved as {output_file}")