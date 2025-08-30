# preprocess_kcc.py
import re
import pandas as pd
from pathlib import Path

# --------- Helpers ----------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

# Heuristic intent rules from columns + text
def infer_intent(row) -> str:
    q = (row.get("QueryText") or "") + " " + (row.get("KccQus") or "")
    cat = (row.get("Category") or "") + " " + (row.get("QueryType") or "")
    text = f"{cat} {q}".lower()

    if any(k in text for k in ["seed", "বীজ", "बीज", "seed rate", "variety", "hybrid"]):
        return "seed_recommendation"

    if any(k in text for k in ["fertilizer", "fertiliser", "npk", "urea", "खाद", "সার"]):
        return "fertilizer_management"

    if any(k in text for k in ["pest", "disease", "insect", "blast", "blight", "रोग", "পোকা", "রোগ"]):
        return "pest_disease_issue"

    if any(k in text for k in ["weather", "rain", "temperature", "humidity", "আবহাওয়া", "मौसम"]):
        return "weather_advisory"

    if any(k in text for k in ["scheme", "yojana", "pm-kisan", "kcc", "credit card", "subsidy", "যোজনা"]):
        return "government_scheme"

    if any(k in text for k in ["market", "mandi", "price", "msP", "দাম", "भाव"]):
        return "market_info"

    return "other"

def run(input_path: str = "Dataset_Chatbot.xlsx", output_csv: str = "processed_data.csv"):
    df = pd.read_excel(input_path)

    # Normalize column names expected by pipeline
    # Try best-effort mapping for common variants
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("querytext", "kccqus", "query", "question"):
            rename_map[c] = "QueryText"
        elif lc in ("kccans", "answer", "response"):
            rename_map[c] = "KccAns"
        elif lc == "category":
            rename_map[c] = "Category"
        elif lc == "querytype":
            rename_map[c] = "QueryType"

    df = df.rename(columns=rename_map)

    # Keep only rows with some text and answer
    df["QueryText"] = df.get("QueryText", pd.Series([""] * len(df))).apply(clean_text)
    df["KccAns"] = df.get("KccAns", pd.Series([""] * len(df))).apply(clean_text)
    df = df[(df["QueryText"] != "") & (df["KccAns"] != "")].copy()

    # Add intent
    df["intent"] = df.apply(infer_intent, axis=1)

    # Minimal set
    out = df[["QueryText", "KccAns", "Category", "QueryType", "intent"]].reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Preprocessed {len(out)} rows → {output_csv}")

if __name__ == "__main__":
    run()
