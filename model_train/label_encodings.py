import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your preprocessed dataset (CSV not Excel)
df = pd.read_csv("processed_data.csv")
# Make sure columns are: "QueryText", "intent"

# Encode the labels (intent → numerical)
label_encoder = LabelEncoder()
df["intent_label"] = label_encoder.fit_transform(df["intent"])

# Save label mapping (for later inference)
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Train-test split (80-20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["intent_label"])

# Save to CSV
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
