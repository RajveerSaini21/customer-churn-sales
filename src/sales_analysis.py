import matplotlib.pyplot as plt

def monthly_sales(transactions):
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"])
    transactions["month"] = transactions["InvoiceDate"].dt.to_period("M")

    monthly = transactions.groupby("month")["Revenue"].sum()

    monthly.plot()
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.show()
