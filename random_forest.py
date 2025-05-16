# E-Commerce Product Recommendation using Random Forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --------------------------
# STEP 1: Simulate Dummy Data
# --------------------------
n = 1000
np.random.seed(42)

brands = ["BrandA", "BrandB", "BrandC"]
categories = ["Electronics", "Clothing", "Home"]

products = pd.DataFrame({
    "price": np.random.randint(100, 1000, size=n),
    "brand": np.random.choice(brands, size=n),
    "category": np.random.choice(categories, size=n),
    "rating": np.round(np.random.uniform(1.0, 5.0, size=n), 1),
    "user_age": np.random.randint(18, 60, size=n),
    "previous_clicks": np.random.randint(0, 50, size=n),
    "user_likes": np.random.choice([0, 1], size=n, p=[0.4, 0.6])
})

# --------------------------
# STEP 2: Preprocessing
# --------------------------
le_brand = LabelEncoder()
le_cat = LabelEncoder()

products["brand"] = le_brand.fit_transform(products["brand"])
products["category"] = le_cat.fit_transform(products["category"])

X = products.drop("user_likes", axis=1)
y = products["user_likes"]

# --------------------------
# STEP 3: Train-Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# STEP 4: Train Random Forest
# --------------------------
rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

# --------------------------
# STEP 5: Evaluation
# --------------------------
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --------------------------
# STEP 6: Feature Importance
# --------------------------
feat_importance = pd.Series(rf.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', title="Feature Importance")
plt.tight_layout()
plt.show()

# --------------------------
# STEP 7: Make a prediction for a new product
# --------------------------
sample_product = pd.DataFrame({
    "price": [499],
    "brand": le_brand.transform(["BrandA"]),
    "category": le_cat.transform(["Electronics"]),
    "rating": [4.2],
    "user_age": [25],
    "previous_clicks": [10]
})

prediction = rf.predict(sample_product)
print("\nSample product prediction (1 = like, 0 = not like):", prediction[0])
