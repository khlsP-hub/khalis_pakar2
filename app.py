from flask import Flask, render_template, request
import os, json, joblib
import numpy as np
import re

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "model")

# ====== Label field (Bahasa Indonesia) ======
FIELD_LABELS = {
    "Perceived_Hearing_Meaning": "Makna Pendengaran",
    "Hearing_FOMO": "FOMO Terkait Pendengaran",
    "Hearing_Test_Barrier": "Hambatan Tes Pendengaran",
    "Missed_Important_Sounds": "Melewatkan Suara Penting",
    "Left_Out_Due_To_Hearing": "Merasa Tertinggal karena Pendengaran",
    "Belief_Early_Hearing_Care": "Keyakinan Perawatan Dini",
    "Last_Hearing_Test_Method": "Metode Tes Pendengaran Terakhir",
    "Interest_in_Hearing_App": "Minat pada Aplikasi Pendengaran",
    "Desired_App_Features": "Fitur Aplikasi yang Diinginkan",
    "Awareness_on_hearing_and_Willingness_to_invest": "Kesadaran & Kemauan Berinvestasi",
    "Paid_App_Test_Interest": "Minat Membayar Tes di Aplikasi",
    "Age_group": "Kelompok Usia",
    "Ear_Discomfort_After_Use": "Ketidaknyamanan Telinga Setelah Pemakaian",
    "Daily_Headphone_Use": "Durasi Pemakaian Headphone",
}

# ====== Pilihan disederhanakan (value = kategori asli!) ======
UI_CHOICES = {
    "Ear_Discomfort_After_Use": [
        ("Never", "Tidak Pernah"),
        ("Sometimes", "Kadang-kadang"),
        ("Often", "Sering / Selalu"),  # gabung Always -> Often
    ],
    "Missed_Important_Sounds": [
        ("No", "Tidak"),
        ("Yes", "Ya"),
    ],
    "Left_Out_Due_To_Hearing": [
        ("No", "Tidak"),
        ("Yes", "Ya"),
    ],
    "Daily_Headphone_Use": [
        ("<1 hour", "< 1 jam"),
        ("1-2 hours", "1–2 jam"),
        ("3-4 hours", "3–4 jam"),
        ("More than 4 hours", "> 4 jam"),
    ],
    "Hearing_Test_Barrier": [
        ("None", "Tidak Ada"),
        ("Fear", "Takut / Malu"),
        ("Cost", "Biaya"),
    ],
    "Last_Hearing_Test_Method": [
        ("At a hospital or clinic", "Di rumah sakit/klinik"),
        ("At home", "Di rumah"),
    ],
    "Hearing_FOMO": [
        ("Never", "Tidak Pernah"),
        ("Sometimes", "Kadang-kadang"),
        ("Often", "Sering / Selalu"),
    ],
    "Interest_in_Hearing_App": [
        ("Yes", "Ya"),
        ("Maybe", "Mungkin"),
        ("No", "Tidak"),
    ],
    "Paid_App_Test_Interest": [
        ("Yes", "Ya"),
        ("Maybe", "Mungkin"),
        ("No", "Tidak"),
    ],
    "Age_group": [
        ("Child", "Anak"),
        ("Teen", "Remaja"),
        ("Adult", "Dewasa"),
        ("Older adult", "Lansia"),
    ],
    "Perceived_Hearing_Meaning": [
        ("Enjoying music, laughter, and life", "Menikmati musik & momen"),
        ("Communication and social connection", "Komunikasi & koneksi sosial"),
        ("Safety and awareness", "Keamanan & kewaspadaan"),
    ],
    "Desired_App_Features": [
        ("Audio amplifier or sound booster", "Penguat suara"),
        ("Hearing test and monitoring", "Tes & pemantauan pendengaran"),
        ("Noise reduction and clarity", "Reduksi bising / kejernihan"),
    ],
    "Awareness_on_hearing_and_Willingness_to_invest": [
        ("Low", "Rendah"),
        ("Medium", "Sedang"),
        ("High", "Tinggi"),
    ],
}

# ====== Humanize kategori (termasuk interval binning) ======
def _fmt_num(x: str):
    try:
        f = float(x)
        return str(int(f)) if abs(f - int(f)) < 1e-9 else str(f)
    except Exception:
        return x

def humanize_category_text(feat_name: str, c: str) -> str:
    M = {
        "<1 hour": "< 1 jam",
        "1-2 hours": "1–2 jam",
        "2-4 hours": "2–4 jam",
        "3-4 hours": "3–4 jam",
        "More than 4 hours": "> 4 jam",
        "Never": "Tidak Pernah",
        "Sometimes": "Kadang-kadang",
        "Often": "Sering",
        "Always": "Selalu",
        "No": "Tidak",
        "Yes": "Ya",
        "Maybe": "Mungkin",
        "At a hospital or clinic": "Di rumah sakit/klinik",
        "At home": "Di rumah",
        "Child": "Anak",
        "Teen": "Remaja",
        "Adult": "Dewasa",
        "Older adult": "Lansia",
        "Low": "Rendah",
        "Medium": "Sedang",
        "High": "Tinggi",
        "Audio amplifier or sound booster": "Penguat suara",
        "Hearing test and monitoring": "Tes & pemantauan pendengaran",
        "Noise reduction and clarity": "Reduksi bising / kejernihan",
        "Communication and social connection": "Komunikasi & koneksi sosial",
        "Safety and awareness": "Keamanan & kewaspadaan",
        "Enjoying music, laughter, and life": "Menikmati musik & momen",
    }
    if c in M:
        return M[c]

    # interval binning: "(0.999, 3.0]" dll
    m = re.match(r"^[\(\[]\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*[\)\]]$", c)
    if m:
        a, b = _fmt_num(m.group(1)), _fmt_num(m.group(2))
        left_open = c.strip().startswith("(")
        right_close = c.strip().endswith("]")
        left = f">{a}" if left_open else f"≥{a}"
        right = f"≤{b}" if right_close else f"<{b}"
        return f"{left} sampai {right}"

    return c

# ====== Util: cari key "mirip" & kanonisasi nilai kategori ======
def _find_key(row: dict, *keywords) -> str:
    # cari key yang mengandung semua keywords (case-insensitive, regex)
    for k in row.keys():
        s = k.lower()
        if all(re.search(kw, s) for kw in keywords):
            return k
    return ""

def _canon_match(v: str, cats: list[str]):
    """
    Cocokkan v ke salah satu kategori di cats dengan normalisasi ringan:
    - lowercase
    - kompres spasi
    - hapus slash/whitespace untuk kasus 'Sering / Selalu' vs 'Sering/selalu'
    - samakan en-dash '–' dan hyphen '-'
    """
    if v is None:
        return None
    v0 = str(v)
    v_clean = re.sub(r"\s+", " ", v0.replace("–", "-")).strip().lower()
    v_comp  = re.sub(r"[\s/]+", "", v_clean)
    for c in cats:
        c_clean = re.sub(r"\s+", " ", str(c).replace("–", "-")).strip().lower()
        if v_clean == c_clean:
            return c
    for c in cats:
        c_comp = re.sub(r"[\s/]+", "", str(c).replace("–", "-")).strip().lower()
        if v_comp == c_comp:
            return c
    return None

# ====== Heuristik risiko berbasis input mentah (bukan nilai fallback) ======
def assess_risk_from_input(row: dict) -> tuple[str, str]:
    def norm(x: str) -> str:
        return str(x or "").strip().lower()

    def is_yes(x: str) -> bool:
        x = norm(x)
        return x in {"yes", "ya", "iya"} or x.startswith("yes ") or x.startswith("ya ") or x.startswith("iya ")

    def is_no(x: str) -> bool:
        x = norm(x)
        return x in {"no", "tidak", "ga", "gak", "nggak", "tdk"} or x.startswith("no ") or x.startswith("tidak ")

    def is_often(x: str) -> bool:
        x = norm(x)
        return any(tok in x for tok in ["often", "always", "sering", "selalu", "sering / selalu", "sering/selalu"])

    def is_sometimes(x: str) -> bool:
        x = norm(x)
        return any(tok in x for tok in ["sometimes", "occasionally", "kadang", "kadang-kadang"])

    def hp_gt4(x: str) -> bool:
        x = norm(x)
        return ("more than 4" in x) or ("> 4" in x) or (">4" in x) or ("lebih dari 4" in x)

    def hp_3_4(x: str) -> bool:
        x = norm(x)
        return ("3-4" in x) or ("3–4" in x)

    def barrier_high(x: str) -> bool:
        x = norm(x)
        return any(tok in x for tok in ["fear", "takut", "malu", "cost", "biaya"])

    # cari key yang cocok meski namanya beda-beda
    k_discomfort = _find_key(row, r"discomfort")
    k_missed     = _find_key(row, r"missed", r"sound")
    k_leftout    = _find_key(row, r"left", r"hear")
    k_headphone  = _find_key(row, r"headphone|daily.*use|use")
    k_barrier    = _find_key(row, r"barrier|hambatan")

    discomfort = row.get(k_discomfort, "")
    missed     = row.get(k_missed, "")
    left_out   = row.get(k_leftout, "")
    headphone  = row.get(k_headphone, "")
    barrier    = row.get(k_barrier, "")

    reasons, high_flags, med_flags = [], 0, 0

    if is_often(discomfort):
        reasons.append("Sering/selalu tidak nyaman setelah penggunaan")
        high_flags += 1
    elif is_sometimes(discomfort):
        reasons.append("Kadang-kadang tidak nyaman setelah penggunaan")
        med_flags += 1

    if is_yes(missed):
        reasons.append("Sering melewatkan suara penting")
        high_flags += 1

    if is_yes(left_out):
        reasons.append("Sering merasa tertinggal karena pendengaran")
        high_flags += 1

    if hp_gt4(headphone):
        reasons.append("Durasi headphone tinggi (> 4 jam/hari)")
        high_flags += 1
    elif hp_3_4(headphone):
        reasons.append("Durasi headphone sedang (3–4 jam/hari)")
        med_flags += 1

    if barrier_high(barrier):
        reasons.append("Ada hambatan untuk tes pendengaran")
        med_flags += 1

    if high_flags >= 2:
        return "Rusak", "; ".join(dict.fromkeys(reasons))
    elif high_flags == 1 or med_flags >= 2:
        return "Potensi Rusak", "; ".join(dict.fromkeys(reasons))
    else:
        return "Sehat", "; ".join(dict.fromkeys(reasons)) if reasons else ""

# ====== Load artifacts ======
def load_artifacts():
    schema = json.load(open(os.path.join(MODEL_DIR, "schema.json"), encoding="utf-8"))
    enc    = joblib.load(os.path.join(MODEL_DIR, "encoder.joblib"))
    model  = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
    metrics = {}
    mp = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(mp):
        metrics = json.load(open(mp, encoding="utf-8"))
    return schema, enc, model, metrics

# ====== Routes ======
@app.route("/", methods=["GET"])
def index():
    schema, _, _, metrics = load_artifacts()

    # Validator: cek kecocokan UI_CHOICES vs schema
    for feat in schema["features"]:
        name = feat["name"]
        cats = set(re.sub(r"\s+", " ", str(c).replace("–", "-")).strip().lower() for c in feat.get("categories", []))
        if name in UI_CHOICES:
            ui_vals = set(re.sub(r"[\s/]+", "", re.sub(r"\s+", " ", v).replace("–", "-")).strip().lower()
                          for v, _ in UI_CHOICES[name])
            cats_comp = set(re.sub(r"[\s/]+", "", c) for c in cats)
            missing = ui_vals - cats_comp
            if missing:
                print(f"[WARN] UI_CHOICES untuk '{name}' mengandung value yang tidak ada di schema.json (setelah kanonisasi): {missing}")

    # Humanized category labels untuk semua fitur
    category_labels = {}
    for feat in schema["features"]:
        name = feat["name"]
        labels_map = {}
        for c in feat.get("categories", []):
            labels_map[c] = humanize_category_text(name, c)
        category_labels[name] = labels_map

    return render_template(
        "index.html",
        schema=schema,
        metrics=metrics,
        ui_choices=UI_CHOICES,
        field_labels=FIELD_LABELS,
        category_labels=category_labels
    )

@app.route("/predict", methods=["POST"])
def predict():
    schema, enc, model, metrics = load_artifacts()

    row_values, row_dict = [], []
    # gunakan list of tuples agar kita bisa lihat (name, original, matched)
    for feat in schema["features"]:
        name = feat["name"]
        cats = feat.get("categories", []) or []

        # ambil dari form (as-is)
        form_v = request.form.get(name, "")
        original_v = form_v if (form_v is not None) else ""

        # cari match kanonis untuk keperluan encoder
        match = _canon_match(original_v, cats)

        # untuk model (row_values): pastikan selalu valid kategori encoder
        if match is not None:
            model_v = match
        else:
            # kalau kosong dan ada kategori, fallback ke cats[0] HANYA untuk model
            model_v = cats[0] if (not original_v and cats) else (original_v if original_v in cats else (cats[0] if cats else ""))

        # untuk heuristik (row_dict): simpan nilai ORIGINAL, jangan dipaksa ke cats[0]
        row_dict.append((name, original_v))

        row_values.append(model_v)

    # ubah row_dict ke dict
    inp_dict = {k: v for k, v in row_dict}

    # DEBUG optional
    print("[PREDICT original row_dict]:", inp_dict)
    print("[PREDICT encoded row_values]:", row_values)

    # Encode & prediksi
    X = enc.transform(np.array([row_values], dtype=object))
    y_pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)[0]
        proba = list(zip(model.classes_, p))

    risk_level, risk_reasons = assess_risk_from_input(inp_dict)

    # Build category_labels ulang (agar result.html bisa render label Indo)
    category_labels = {
        f["name"]: {c: humanize_category_text(f["name"], c) for c in f.get("categories", [])}
        for f in schema["features"]
    }

    return render_template(
        "result.html",
        schema=schema,
        inp=inp_dict,
        prediction=y_pred,
        proba=proba,
        metrics=metrics,
        risk_level=risk_level,
        risk_reasons=risk_reasons,
        field_labels=FIELD_LABELS,
        category_labels=category_labels
    )

if __name__ == "__main__":
    app.run(debug=True)