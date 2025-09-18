import os, json, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# === SETUP DI SINI ===
DATA_FILE = "Hearing_wellness_extended.csv"   # nama file dataset
TARGET_COL = "Ear_Discomfort_After_Use"
TARGET_COL = "Missed_Important_Sounds"
TARGET_COL = "Daily_Headphone_Use"

def read_any(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def bin_numeric_series(s, n_bins=4):
    try:
        cats, bins = pd.qcut(s, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        cats, bins = pd.cut(s, bins=n_bins, duplicates="drop")
    return cats.astype(str), bins.tolist()

def build_schema(df, target):
    schema = {"target": target, "features": []}
    for col in df.columns:
        if col == target:
            continue
        info = {"name": col, "type": None, "categories": None, "bins": None}
        if pd.api.types.is_numeric_dtype(df[col]):
            binned, bins = bin_numeric_series(df[col])
            df[col] = binned
            info["type"] = "numeric_binned"
            info["bins"] = bins
            info["categories"] = sorted(df[col].dropna().unique().tolist())
        else:
            df[col] = df[col].astype(str)
            info["type"] = "categorical"
            info["categories"] = sorted(df[col].dropna().astype(str).unique().tolist())
        schema["features"].append(info)
    return df, schema

def impute(df, target):
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).replace({"nan": np.nan})
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

def main():
    df = read_any(DATA_FILE)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Kolom target '{TARGET_COL}' tidak ditemukan. Kolom yang ada: {list(df.columns)}")

    df = impute(df, TARGET_COL)
    Xy, schema = build_schema(df.copy(), TARGET_COL)
    y = Xy[TARGET_COL].astype(str)
    X = Xy.drop(columns=[TARGET_COL])

    # encode semua fitur
    categories = [feat["categories"] for feat in schema["features"]]
    enc = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = enc.fit_transform(X)

    # train dengan semua data
    clf = CategoricalNB()
    clf.fit(X_enc, y)

    # evaluasi training accuracy
    yhat = clf.predict(X_enc)
    acc = accuracy_score(y, yhat)
    report = classification_report(y, yhat, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, yhat)

    # simpan artefak
    os.makedirs("model", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    joblib.dump(clf, "model/model.joblib")
    joblib.dump(enc, "model/encoder.joblib")
    json.dump(schema, open("model/schema.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump({"accuracy": acc, "report": report}, 
              open("model/metrics.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    # plot confusion matrix
    fig = plt.figure(figsize=(5,4), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Training)")
    plt.colorbar()
    classes = sorted(y.unique())
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png")
    plt.close(fig)

    print(f"Akurasi (training): {acc:.4f}")
    print("Artefak tersimpan di folder 'model/' dan 'static/'")

if __name__ == "__main__":
    main()
