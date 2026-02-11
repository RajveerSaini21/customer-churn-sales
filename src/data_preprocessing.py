import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- CUSTOMER DATA ---------------- #
def preprocess_customer_data(input_path, output_path):
    df = pd.read_csv(input_path)

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Separate target
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # Encode categorical features
    label_cols = X.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in label_cols:
        X[col] = le.fit_transform(X[col])

    # Scale ONLY features
    scaler = StandardScaler()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # Reattach target
    X["Churn"] = y.values

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    X.to_csv(output_path, index=False)

    print("Customer data saved to:", os.path.abspath(output_path))
    return X


# ---------------- TRANSACTION DATA ---------------- #
def preprocess_transaction_data(input_path, output_path):
    df = pd.read_csv(input_path)

    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    df.dropna(inplace=True)

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")

    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Transaction data saved to:", os.path.abspath(output_path))
    return df


# ---------------- CUSTOMER SEGMENTATION ---------------- #
def create_customer_segments(customer_df, output_path, n_clusters=4):
    """
    Creates customer segments using KMeans clustering
    """

    X = customer_df.drop("Churn", axis=1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    customer_df["Segment"] = kmeans.fit_predict(X)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer_df.to_csv(output_path, index=False)

    print("Customer segments saved to:", os.path.abspath(output_path))
    return customer_df


# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":

    customers_cleaned = preprocess_customer_data(
        os.path.join(BASE_DIR, "data", "raw", "customer_churn.csv"),
        os.path.join(BASE_DIR, "data", "processed", "customers_cleaned.csv")
    )

    preprocess_transaction_data(
        os.path.join(BASE_DIR, "data", "raw", "transactions.csv"),
        os.path.join(BASE_DIR, "data", "processed", "transactions_cleaned.csv")
    )

    create_customer_segments(
        customers_cleaned,
        os.path.join(BASE_DIR, "data", "processed", "customer_segments.csv")
    )
