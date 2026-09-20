import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop the identifier and the leftover row-index column: neither predicts purchase
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True, errors="ignore")

# Fix inconsistent label
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Categorical columns stay as raw strings; the training pipeline one-hot-encodes them,
# and the Streamlit app sends raw category values, so training and serving stay consistent.
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# stratify keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Train rows: {len(Xtrain)}, Test rows: {len(Xtest)}")
print("ProdTaken distribution in train:")
print(ytrain.value_counts())
