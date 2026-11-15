
import pandas as pd

# CUSTOMER CHURN VISUALIZATION

import matplotlib.pyplot as plt
import seaborn as sns
import os

import numpy as np

# Load the customer churn dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")

"""
# Display first five rows
print("\nFIRST FIVE ROWS")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())



#check data types and non null counts
print("\n dataset information")
df.info()


#check the missing values
print("\n missing values")
print(df.isnull().sum())

#check the duplicate values
print("\n duplicate rows")
print(df.duplicated().sum())

#check churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())

# 5. Check churn percentages
print("\nCHURN PERCENTAGE")
print(df["Churn"].value_counts(normalize=True) * 100)

"""

# Investigate TotalCharges

print("\nTOTALCHARGES DATA TYPE")
print(df["TotalCharges"].dtype)

# Count empty or whitespace-only values
blank_count = df["TotalCharges"].str.strip().eq("").sum()

print("\nBLANK TOTALCHARGES")
print("Number of blank values:", blank_count)

# Convert temporarily to numeric to identify invalid values
total_charges_numeric = pd.to_numeric(
    df["TotalCharges"].str.strip(),
    errors="coerce"
)


print("\nINVALID NUMERIC VALUES")
print(total_charges_numeric.isna().sum())

# Display records with blank TotalCharges
print("\nCUSTOMERS WITH BLANK TOTALCHARGES")
print(
    df.loc[
        total_charges_numeric.isna(),
        ["customerID", "tenure", "TotalCharges"]
    ]
)


# Create a separate cleaned dataset
df_clean = df.copy()

# Convert TotalCharges into numeric values
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"].str.strip(),
    errors="coerce"
)

# Replace missing TotalCharges with 0
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0)

# Verify the cleaning
print("\nCLEANED DATASET INFORMATION")
print(df_clean.info())

print("\nMISSING VALUES AFTER CLEANING")
print(df_clean.isnull().sum())

print("\nTOTALCHARGES DATA TYPE")
print(df_clean["TotalCharges"].dtype)

# Save cleaned dataset
df_clean.to_csv(
    "outputs/Telco-Customer-Churn-Cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")



# EXPLORATORY DATA ANALYSIS

# 1. Numerical summary
print("\nNUMERICAL SUMMARY")
print(
    df_clean[
        ["tenure", "MonthlyCharges", "TotalCharges"]
    ].describe()
)

# 2. Categorical summary
print("\nCATEGORICAL SUMMARY")
print(
    df_clean[
        ["gender", "Contract", "InternetService", "PaymentMethod", "Churn"]
    ].describe()
)

# 3. Unique values in categorical columns
print("\nUNIQUE CATEGORY VALUES")

categorical_columns = df_clean.select_dtypes(
    include=["str", "object"]
).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df_clean[column].value_counts())


# CHURN RATE ANALYSIS

def churn_analysis(column):

    result = df_clean.groupby(column).agg(
        Total_Customers=("Churn", "count"),
        Churned_Customers=(
            "Churn",
            lambda x: (x == "Yes").sum()
        )
    )

    result["Churn_Rate_%"] = (
        result["Churned_Customers"]
        / result["Total_Customers"]
        * 100
    )

    print(f"\nCHURN ANALYSIS BY {column.upper()}")
    print(result.round(2))


# Analyze different customer categories

churn_analysis("Contract")
churn_analysis("InternetService")
churn_analysis("PaymentMethod")
churn_analysis("OnlineSecurity")
churn_analysis("TechSupport")




# Create visualization folder
os.makedirs("outputs/charts", exist_ok=True)

sns.set_theme(style="whitegrid")

columns = [
    "Contract",
    "InternetService",
    "PaymentMethod",
    "TechSupport"
]

for column in columns:

    churn_rate = (
        df_clean.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    plt.figure(figsize=(9, 5))

    ax = sns.barplot(
        data=churn_rate,
        x=column,
        y="Churn Rate"
    )

    plt.title(f"Customer Churn Rate by {column}")
    plt.xlabel(column)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=15)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()

    filename = f"outputs/charts/{column}_Churn_Rate.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.close()

    print(f"Chart saved: {filename}")

print("\nAll churn charts generated successfully!")


# CHURN ANALYSIS BY TENURE AND MONTHLY CHARGES

# 1. Group customers by tenure
df_clean["Tenure_Group"] = pd.cut(
    df_clean["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"]
)

# 2. Group customers by monthly charges
df_clean["MonthlyCharges_Group"] = pd.cut(
    df_clean["MonthlyCharges"],
    bins=[0, 30, 60, 90, 120],
    labels=["0-30", "31-60", "61-90", "91-120"],
    include_lowest=True
)

# 3. Create a reusable churn analysis function
def numerical_churn_analysis(column):

    result = df_clean.groupby(column, observed=False).agg(
        Total_Customers=("Churn", "count"),
        Churned_Customers=(
            "Churn", lambda x: (x == "Yes").sum()
        )
    )

    result["Churn_Rate_%"] = (
        result["Churned_Customers"] /
        result["Total_Customers"] * 100
    )

    print(f"\nCHURN ANALYSIS BY {column.upper()}")
    print(result.round(2))


# 4. Display both analyses
numerical_churn_analysis("Tenure_Group")
numerical_churn_analysis("MonthlyCharges_Group")


# TENURE AND MONTHLY CHARGES CHURN CHARTS

columns = ["Tenure_Group", "MonthlyCharges_Group"]

for column in columns:

    churn_rate = (
        df_clean.groupby(column, observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .reset_index(name="Churn Rate")
    )

    plt.figure(figsize=(9, 5))

    ax = sns.barplot(
        data=churn_rate,
        x=column,
        y="Churn Rate"
    )

    plt.title(f"Customer Churn Rate by {column}")
    plt.xlabel(column.replace("_", " "))
    plt.ylabel("Churn Rate (%)")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()

    filename = f"outputs/charts/{column}_Churn_Rate.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart saved: {filename}")

print("\nAll tenure and monthly charges charts generated!")



# MACHINE LEARNING - DATA PREPARATION

from sklearn.model_selection import train_test_split

# Remove customer ID and EDA-only grouping columns
X = df_clean.drop(
    columns=[
        "customerID",
        "Churn",
        "Tenure_Group",
        "MonthlyCharges_Group"
    ]
)

# Convert target into numerical values
# No = 0, Yes = 1
y = df_clean["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Display results
print("\nMACHINE LEARNING DATA PREPARATION")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nData splitting completed successfully!")



# CATEGORICAL ENCODING AND FEATURE SCALING

from sklearn.preprocessing import StandardScaler

# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNUMERICAL FEATURES:")
print(numerical_features)

print("\nCATEGORICAL FEATURES:")
print(categorical_features)

# STEP 1: Apply One-Hot Encoding
X_train_encoded = pd.get_dummies(
    X_train,
    columns=categorical_features,
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=categorical_features,
    dtype=int
)

# Ensure both datasets have identical columns
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

# STEP 2: Apply Feature Scaling
scaler = StandardScaler()

X_train_encoded[numerical_features] = scaler.fit_transform(
    X_train_encoded[numerical_features]
)

X_test_encoded[numerical_features] = scaler.transform(
    X_test_encoded[numerical_features]
)

# STEP 3: Display results
print("\nENCODING AND SCALING RESULTS")

print("Original training shape:", X_train.shape)
print("Processed training shape:", X_train_encoded.shape)

print("Original testing shape:", X_test.shape)
print("Processed testing shape:", X_test_encoded.shape)

print("\nCategorical encoding and scaling completed successfully!")



# LOGISTIC REGRESSION USING NUMPY


# Convert data into NumPy arrays
X_train_np = X_train_encoded.to_numpy(dtype=float)
X_test_np = X_test_encoded.to_numpy(dtype=float)

y_train_np = y_train.to_numpy(dtype=float)

# Initialize model parameters
weights = np.zeros(X_train_np.shape[1])
bias = 0.0

# Training settings
learning_rate = 0.1
epochs = 2000
m = len(y_train_np)

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Train the Logistic Regression model
for epoch in range(epochs):

    # Calculate predictions
    linear_output = np.dot(X_train_np, weights) + bias
    predictions = sigmoid(linear_output)

    # Calculate errors
    errors = predictions - y_train_np

    # Calculate gradients
    dw = (np.dot(X_train_np.T, errors) / m) + (0.01 / m) * weights
    db = np.sum(errors) / m

    # Update weights and bias
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Predict on testing data
test_probabilities = sigmoid(
    np.dot(X_test_np, weights) + bias
)

y_pred = (test_probabilities >= 0.25).astype(int)

print("\nLOGISTIC REGRESSION MODEL")
print("Model training completed successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.head(10).tolist())

print("\nFirst 10 Predicted Values:")
print(y_pred[:10].tolist())

print("\nFirst 10 Churn Probabilities:")
print(np.round(test_probabilities[:10], 3))


# LOGISTIC REGRESSION MODEL EVALUATION

# Actual and predicted values
actual = y_test.to_numpy(dtype=int)
predicted = y_pred.astype(int)

# Calculate confusion matrix values
TP = np.sum((actual == 1) & (predicted == 1))
TN = np.sum((actual == 0) & (predicted == 0))
FP = np.sum((actual == 0) & (predicted == 1))
FN = np.sum((actual == 1) & (predicted == 0))

# Calculate evaluation metrics
accuracy = (TP + TN) / len(actual)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

# Display results
print("\nMODEL EVALUATION RESULTS")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1_score * 100:.2f}%")

print("\nCONFUSION MATRIX")
print("                 Predicted No   Predicted Yes")
print(f"Actual No        {TN:13d}   {FP:13d}")
print(f"Actual Yes       {FN:13d}   {TP:13d}")



# ROC CURVE AND ROC-AUC EVALUATION

# Sort customers by predicted churn probability

sorted_indices = np.argsort(-test_probabilities)

actual_sorted = actual[sorted_indices]
probabilities_sorted = test_probabilities[sorted_indices]

# Calculate True Positive Rate and False Positive Rate
total_positive = np.sum(actual == 1)
total_negative = np.sum(actual == 0)

TPR = []
FPR = []

tp = 0
fp = 0

TPR.append(0)
FPR.append(0)

for i in range(len(actual_sorted)):
    if actual_sorted[i] == 1:
        tp += 1
    else:
        fp += 1

    # Add a point when the next probability is different,
    # so tied scores are handled together.
    if (
        i == len(actual_sorted) - 1
        or probabilities_sorted[i] != probabilities_sorted[i + 1]
    ):
        TPR.append(tp / total_positive)
        FPR.append(fp / total_negative)

# Calculate ROC-AUC using trapezoidal area
roc_auc = np.trapezoid(TPR, FPR)

print("\nROC-AUC EVALUATION")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print(f"ROC-AUC Percentage: {roc_auc * 100:.2f}%")

# Plot ROC Curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot(
    FPR,
    TPR,
    label=f"Logistic Regression (AUC = {roc_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Customer Churn Prediction")

plt.legend()
plt.grid(True)

plt.savefig(
    "outputs/charts/Logistic_Regression_ROC_Curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nROC Curve saved successfully!")

# DECISION TREE - NUMPY CHECK

print("\nDECISION TREE DATA CHECK")

print("Training data:", X_train_encoded.shape)
print("Testing data:", X_test_encoded.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

print("\nDecision Tree preparation successful!")


# DECISION TREE - GINI IMPURITY

def calculate_gini(y):

    if len(y) == 0:
        return 0

    probabilities = np.bincount(y, minlength=2) / len(y)

    gini = 1 - np.sum(probabilities ** 2)

    return gini


# Test Gini Impurity

sample = np.array([0, 0, 0, 1, 1])

gini_value = calculate_gini(sample)

print("\nGINI IMPURITY TEST")
print("Sample values:", sample)
print("Gini Impurity:", round(gini_value, 4))


# DECISION TREE - FIND BEST SPLIT

def find_best_split(X, y):

    best_gain = 0
    best_feature = None
    best_threshold = None

    parent_gini = calculate_gini(y)

    for feature in range(X.shape[1]):

        values = np.unique(X[:, feature])

        for threshold in values:

            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_y = y[left_mask]
            right_y = y[right_mask]

            weighted_gini = (
                len(left_y) / len(y) * calculate_gini(left_y)
                + len(right_y) / len(y) * calculate_gini(right_y)
            )

            gain = parent_gini - weighted_gini

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain


# Test best split using training data

X_train_np = X_train_encoded.to_numpy(dtype=float)
y_train_np = y_train.to_numpy(dtype=int)

feature, threshold, gain = find_best_split(
    X_train_np,
    y_train_np
)

print("\nBEST DECISION TREE SPLIT")
print("Best feature index:", feature)
print("Best threshold:", threshold)
print("Information gain:", round(gain, 4))

if feature is not None:
    print("Best feature name:", X_train_encoded.columns[feature])


# DECISION TREE CLASSIFIER - BUILD TREE

def build_tree(X, y, depth=0, max_depth=5, min_samples=20):

    # Count customers in each class
    class_counts = np.bincount(y, minlength=2)

    # Majority class prediction
    prediction = np.argmax(class_counts)

    # Stop conditions
    if (
        depth >= max_depth
        or len(y) < min_samples
        or calculate_gini(y) == 0
    ):
        return {"prediction": int(prediction)}

    # Find best split
    feature, threshold, gain = find_best_split(X, y)

    # Stop if no useful split exists
    if feature is None or gain <= 0:
        return {"prediction": int(prediction)}

    # Divide data
    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    # Build child branches
    left_tree = build_tree(
        X[left_mask],
        y[left_mask],
        depth + 1,
        max_depth,
        min_samples
    )

    right_tree = build_tree(
        X[right_mask],
        y[right_mask],
        depth + 1,
        max_depth,
        min_samples
    )

    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_tree,
        "right": right_tree,
        "prediction": int(prediction)
    }


# Train Decision Tree

decision_tree = build_tree(
    X_train_np,
    y_train_np,
    max_depth=5,
    min_samples=20
)

print("\nDECISION TREE TRAINING")
print("Decision Tree trained successfully!")
print("Maximum tree depth: 5")
print("Minimum samples per leaf: 20")
print("Training customers:", len(y_train_np))

print("Maximum tree depth: 5")
print("Minimum samples per leaf: 20")
print("Training customers:", len(y_train_np))


# DECISION TREE - PREDICTION FUNCTION

def predict_tree(tree, sample):

    # Check whether current node is a leaf
    if "feature" not in tree:
        return tree["prediction"]

    feature = tree["feature"]
    threshold = tree["threshold"]

    # Follow the appropriate branch
    if sample[feature] <= threshold:
        return predict_tree(tree["left"], sample)

    else:
        return predict_tree(tree["right"], sample)


# Predict testing data

tree_predictions = np.array([
    predict_tree(decision_tree, sample)
    for sample in X_test_np
])

print("\nDECISION TREE PREDICTIONS")
print("Prediction completed successfully!")

print("\nFirst 10 Actual Values:")
print(y_test.to_numpy()[:10].tolist())

print("\nFirst 10 Predicted Values:")
print(tree_predictions[:10].tolist())

print("\nTotal predictions:", len(tree_predictions))



# DECISION TREE MODEL EVALUATION

tree_actual = y_test.to_numpy(dtype=int)
tree_predicted = tree_predictions.astype(int)

# Calculate confusion matrix values
TP = np.sum((tree_actual == 1) & (tree_predicted == 1))
TN = np.sum((tree_actual == 0) & (tree_predicted == 0))
FP = np.sum((tree_actual == 0) & (tree_predicted == 1))
FN = np.sum((tree_actual == 1) & (tree_predicted == 0))

# Calculate evaluation metrics
accuracy = (TP + TN) / len(tree_actual)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

# Display results
print("\nDECISION TREE EVALUATION RESULTS")

print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1_score * 100:.2f}%")

print("\nCONFUSION MATRIX")
print("                 Predicted No   Predicted Yes")
print(f"Actual No        {TN:13d}   {FP:13d}")
print(f"Actual Yes       {FN:13d}   {TP:13d}")


def build_tree(X, y, depth=0, max_depth=5, min_samples=20):

    class_counts = np.bincount(y, minlength=2)
    prediction = np.argmax(class_counts)

    if depth >= max_depth or len(y) < min_samples or calculate_gini(y) == 0:
        return {"prediction": int(prediction)}

    feature, threshold, gain = find_best_split(X, y)

    if feature is None or gain <= 0:
        return {"prediction": int(prediction)}

    left_mask = X[:, feature] <= threshold
    right_mask = X[:, feature] > threshold

    left_tree = build_tree(
        X[left_mask], y[left_mask],
        depth + 1, max_depth, min_samples
    )

    right_tree = build_tree(
        X[right_mask], y[right_mask],
        depth + 1, max_depth, min_samples
    )

    return {
        "feature": feature,
        "threshold": threshold,
        "gain": float(gain),
        "samples": len(y),
        "left": left_tree,
        "right": right_tree,
        "prediction": int(prediction)
    }


# Rebuild Decision Tree
decision_tree = build_tree(
    X_train_np,
    y_train_np,
    max_depth=5,
    min_samples=20
)

print("\nDECISION TREE UPDATED SUCCESSFULLY!")
print("Maximum depth: 5")
print("Training samples:", len(y_train_np))


# DECISION TREE FEATURE IMPORTANCE

feature_importances = np.zeros(X_train_np.shape[1])

def collect_importance(tree):

    if "feature" not in tree:
        return

    feature = tree["feature"]
    gain = tree["gain"]
    samples = tree["samples"]

    importance = (samples / len(y_train_np)) * gain

    feature_importances[feature] += importance

    collect_importance(tree["left"])
    collect_importance(tree["right"])


collect_importance(decision_tree)

# Normalize importance scores
total_importance = np.sum(feature_importances)

if total_importance > 0:
    feature_importances = feature_importances / total_importance

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": X_train_encoded.columns,
    "Importance": feature_importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nDECISION TREE FEATURE IMPORTANCE")
print(importance_df.head(15).to_string(index=False))

# TOP 10 FEATURE IMPORTANCE CHART

top_features = importance_df.head(10).sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.title("Top 10 Decision Tree Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    "outputs/charts/Decision_Tree_Feature_Importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance chart saved successfully!")


# SAVE DECISION TREE FEATURE IMPORTANCE

importance_df.to_csv(
    "outputs/Decision_Tree_Feature_Importance.csv",
    index=False
)

print("\nFeature importance data saved successfully!")
