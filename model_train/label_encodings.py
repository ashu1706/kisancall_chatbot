# label_encodings.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import json

df = pd.read_csv("processed_data.csv")

# Encode labels
label_encoder = LabelEncoder()
df["intent_label"] = label_encoder.fit_transform(df["intent"])

# Save label mapping for inference
# Save label mapping for inference
label_mapping = {
    str(k): int(v)   # cast numpy.int64 → Python int
    for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
}

with open("label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

# Train/test split
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["intent_label"]
)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("Label mapping saved to label_mapping.json")
print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
