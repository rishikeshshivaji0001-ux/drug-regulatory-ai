import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("drug_regulatory_classification_dataset.csv")

# Drop missing target
df = df.dropna(subset=['Target_Regulatory_Class'])

# Encode target
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Target_Regulatory_Class'] = le.fit_transform(df['Target_Regulatory_Class'])

X = df.drop('Target_Regulatory_Class', axis=1)
y = df['Target_Regulatory_Class']

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64','float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

# Save entire pipeline
joblib.dump(pipeline, "drug_pipeline.pkl")

print("Pipeline saved successfully.")