import pandas as pd

def create_customer_features(transactions):
    features = transactions.groupby("CustomerID").agg({
        "InvoiceNo": "count",
        "Quantity": "sum",
        "UnitPrice": "mean"
    })

    features.rename(columns={
        "InvoiceNo": "purchase_frequency",
        "Quantity": "total_quantity",
        "UnitPrice": "avg_price"
    }, inplace=True)

    return features
