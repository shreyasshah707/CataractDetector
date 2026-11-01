"""
EDD Streamlit Prototype — Eye Disease Detection Dashboard
Tabs: Metrics | Live Inference | Eval Results | Dataset Explorer
"""

import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EDD - Eye Disease Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR   = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "aptos2019.pth"
EVAL_DIR   = BASE_DIR / "eval_results" / "aptos2019"
ODIR_EVAL  = BASE_DIR / "eval_results" / "odir+rfmid"
STATS_FILE = BASE_DIR / "stats.txt"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="collapsedControl"] { display: none; }
.metric-card {
    background: linear-gradient(135deg, #1e2a4a 0%, #16213e 100%);
    border: 1px solid #2d4a7a;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,120,255,0.2); }
.metric-label { font-size: 0.75rem; font-weight: 600; color: #7fa6d5; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 2rem; font-weight: 700; color: #e8f4fd; line-height: 1.1; }
.metric-delta-pos { font-size: 0.8rem; color: #4ade80; margin-top: 4px; }
.metric-delta-neg { font-size: 0.8rem; color: #f87171; margin-top: 4px; }
.metric-delta-neu { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
.section-header { font-size: 1.25rem; font-weight: 700; color: #e2e8f0; border-left: 4px solid #3b82f6; padding-left: 12px; margin: 24px 0 14px 0; }
.pred-box { background: linear-gradient(135deg, #0c1a2e 0%, #1a2744 100%); border: 2px solid #3b82f6; border-radius: 14px; padding: 28px; text-align: center; }
.pred-class { font-size: 1.8rem; font-weight: 800; }
.pred-conf  { font-size: 1rem; color: #94a3b8; margin-top: 6px; }
.pred-info  { font-size: 0.85rem; color: #64748b; margin-top: 10px; font-style: italic; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 👁️ Eye Disease Detection — EDD Dashboard")
st.caption("Research prototype · Phase 2 & Phase 3 results · Not for clinical use")
st.markdown("---")

# ── helpers ──────────────────────────────────────────────────────────────────
def metric_card(label, value, delta=None, delta_type="neu"):
    delta_html = f'<div class="metric-delta-{delta_type}">{delta}</div>' if delta else ""
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>{delta_html}</div>',
        unsafe_allow_html=True,
    )

def dark_ax(ax):
    ax.set_facecolor("#1e293b")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="#94a3b8")

@st.cache_resource
def load_aptos_model():
    try:
        import torch
        import torchvision.models as tv_models
    except ImportError:
        return None, "PyTorch not installed."
    if not MODEL_PATH.exists():
        return None, f"Model file not found at {MODEL_PATH}"
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = tv_models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)
        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(state, dict):
            state = state.get("model_state_dict", state.get("state_dict", state))
        model.load_state_dict(state, strict=False)
        model.eval().to(device)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_fundus(img):
    import torch
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(img.convert("RGB")).unsqueeze(0)

@st.cache_data
def load_eval_data():
    data = {}
    mf = EVAL_DIR / "summary_metrics.json"
    if mf.exists():
        data["metrics"] = json.loads(mf.read_text())
    rf = EVAL_DIR / "classification_report.csv"
    if rf.exists():
        data["report"] = pd.read_csv(rf, index_col=0)
    pf = EVAL_DIR / "test_predictions.csv"
    if pf.exists():
        data["preds"] = pd.read_csv(pf)
    cm = EVAL_DIR / "confusion_matrix.png"
    if cm.exists():
        data["cm_path"] = str(cm)
    return data

@st.cache_data
def load_stats():
    return STATS_FILE.read_text() if STATS_FILE.exists() else ""

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_metrics, tab_infer, tab_eval, tab_data = st.tabs([
    "📊 Metrics", "🔬 Live Inference", "📈 Eval Results", "📂 Dataset Explorer"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB: METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("## 📊 Performance Metrics — All Phases")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Phase 2 Wk3 Accuracy", "79.78%", "ResNet-50 · APTOS 2019", "neu")
    with c2:
        metric_card("Phase 2 Wk4 Bal.Acc", "62.90%", "EfficientNet-B3 · 6-class", "neu")
    with c3:
        metric_card("Phase 3 Wk5 Bal.Acc", "63.41%", "+0.51pp vs Wk4", "pos")
    with c4:
        metric_card("Phase 3 Wk6 Target", "≥ 85%", "RETFound Ensemble", "neu")

    st.markdown('<div class="section-header">Phase Summary</div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame([
        {"Phase": "P2 Wk3", "Architecture": "ResNet-50", "Datasets": "APTOS 2019", "Classes": 5,  "Bal.Acc": "79.78%", "Macro F1": "0.6324", "Key Change": "Single-disease DR baseline",       "Status": "✅ Done"},
        {"Phase": "P2 Wk4", "Architecture": "EfficientNet-B3", "Datasets": "ODIR+RFMiD", "Classes": 6, "Bal.Acc": "62.90%", "Macro F1": "0.5369", "Key Change": "Multi-disease + class balancing", "Status": "✅ Done"},
        {"Phase": "P3 Wk5", "Architecture": "EffNet-B3 + CLAHE + DropPath", "Datasets": "ODIR+RFMiD", "Classes": 6, "Bal.Acc": "63.41%", "Macro F1": "0.5324", "Key Change": "+11.5% Cataract F1 via Mixup/CLAHE", "Status": "✅ Done"},
        {"Phase": "P3 Wk6", "Architecture": "RETFound ViT-L + EffNetV2-S Ensemble", "Datasets": "All 4 datasets", "Classes": 5, "Bal.Acc": "≥ 85%", "Macro F1": "≥ 0.80", "Key Change": "Foundation model + Focal Loss", "Status": "🔄 In Progress"},
    ])
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">APTOS 2019 Per-Grade Breakdown (ResNet-50)</div>', unsafe_allow_html=True)
    eval_data = load_eval_data()
    if "report" in eval_data:
        df_rep = eval_data["report"]
        grade_rows = df_rep[df_rep.index.str.startswith("Grade")]
        if not grade_rows.empty:
            grade_labels = ["Grade 0\n(No DR)", "Grade 1\n(Mild)", "Grade 2\n(Moderate)", "Grade 3\n(Severe)", "Grade 4\n(Proliferative)"]
            x3 = np.arange(len(grade_rows))
            fig3, ax3 = plt.subplots(figsize=(11, 4), facecolor="#0f172a")
            dark_ax(ax3)
            w = 0.25
            for i, (col, color) in enumerate(zip(["precision", "recall", "f1-score"], ["#60a5fa", "#34d399", "#f472b6"])):
                vals = grade_rows[col].values.astype(float)
                ax3.bar(x3 + (i - 1) * w, vals, w, label=col.title(), color=color, alpha=0.85)
            ax3.set_xticks(x3)
            ax3.set_xticklabels(grade_labels[:len(grade_rows)], color="#e2e8f0")
            ax3.set_ylabel("Score", color="#94a3b8")
            ax3.set_ylim(0, 1.1)
            ax3.set_title("APTOS 2019 DR Grading — Per-Grade Metrics (ResNet-50)", color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
            ax3.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
            st.pyplot(fig3)
            plt.close()
    else:
        st.info("Classification report not found.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    st.markdown("## 🔬 Live Fundus Image Inference")
    st.markdown("> **Model:** ResNet-50 (APTOS 2019 checkpoint) — **Accuracy:** 79.78% — **Task:** DR severity grading (Grade 0–4)")

    GRADE_INFO = {
        0: ("Grade 0 — No DR",          "#22c55e", "No diabetic retinopathy detected. Retina appears healthy."),
        1: ("Grade 1 — Mild NPDR",       "#84cc16", "Mild: Microaneurysms only. Monitor closely."),
        2: ("Grade 2 — Moderate NPDR",   "#eab308", "Moderate: More than microaneurysms. Refer to ophthalmologist."),
        3: ("Grade 3 — Severe NPDR",     "#f97316", "Severe: Many hemorrhages. High risk. Urgent referral."),
        4: ("Grade 4 — Proliferative DR","#ef4444", "PDR: Neovascularization present. Sight-threatening. Urgent treatment."),
    }

    uploaded = st.file_uploader("Upload a fundus photograph (JPG / PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        img = Image.open(uploaded)
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.image(img, caption="Uploaded Fundus Image", use_container_width=True)
        with col_res:
            with st.spinner("Running inference..."):
                model, error = load_aptos_model()
            if error:
                st.error(f"Could not load model: {error}")
            else:
                try:
                    import torch, torch.nn.functional as F_t
                    tensor = preprocess_fundus(img).to(next(model.parameters()).device)
                    with torch.no_grad():
                        probs = F_t.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
                    pred = int(np.argmax(probs))
                    name, color, desc = GRADE_INFO[pred]
                    conf = float(probs[pred]) * 100
                    st.markdown(
                        f'<div class="pred-box">'
                        f'<div class="pred-class" style="color:{color}">{name}</div>'
                        f'<div class="pred-conf">Confidence: {conf:.1f}%</div>'
                        f'<div class="pred-info">{desc}</div></div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown("#### Probability Distribution")
                    fig_p, ax_p = plt.subplots(figsize=(6, 3), facecolor="#0f172a")
                    dark_ax(ax_p)
                    gl = ["Grade 0\nNo DR","Grade 1\nMild","Grade 2\nModerate","Grade 3\nSevere","Grade 4\nProliferative"]
                    gc = ["#22c55e","#84cc16","#eab308","#f97316","#ef4444"]
                    bars_p = ax_p.barh(gl, probs * 100, color=gc, alpha=0.85)
                    ax_p.set_xlabel("Probability (%)", color="#94a3b8")
                    ax_p.set_xlim(0, 100)
                    for b, v in zip(bars_p, probs * 100):
                        ax_p.text(v + 0.5, b.get_y() + b.get_height()/2, f"{v:.1f}%", va="center", color="#e2e8f0", fontsize=8)
                    plt.tight_layout()
                    st.pyplot(fig_p)
                    plt.close()
                except Exception as e:
                    st.error(f"Inference error: {e}")
    else:
        st.info("👆 Upload a fundus image to begin inference.")
        st.markdown("""
| Checkpoint | Architecture | Task | Size |
|---|---|---|---|
| `models/aptos2019.pth` | ResNet-50 | DR Severity Grading (Grade 0–4) | ~270 MB |

> The **RETFound ensemble** (Phase 3 Wk6) checkpoint must be trained on Kaggle first — see Dataset Explorer tab.
""")

    st.markdown("---")
    st.warning("**Clinical Disclaimer:** Research prototype only. Not FDA-cleared or CE-marked. Do not use as a standalone diagnostic tool.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: EVAL RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("## 📈 Evaluation Results")
    eval_data = load_eval_data()

    st.markdown('<div class="section-header">APTOS 2019 — Phase 2 Week 3 (ResNet-50)</div>', unsafe_allow_html=True)
    if "metrics" in eval_data:
        m = eval_data["metrics"]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Test Accuracy", f"{m.get('Accuracy', 0)*100:.2f}%", "ResNet-50 · APTOS 2019")
        with c2:
            metric_card("Macro F1-Score", f"{m.get('Macro F1-Score', 0):.4f}", "5-class DR grading")
        with c3:
            metric_card("Test Samples", "366", "Hold-out test set")

    col_cm, col_rep = st.columns([1, 1])
    with col_cm:
        if "cm_path" in eval_data:
            st.markdown("#### Confusion Matrix")
            st.image(eval_data["cm_path"], use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")
    with col_rep:
        if "report" in eval_data:
            st.markdown("#### Classification Report")
            df_rd = eval_data["report"].copy()
            df_rd[df_rd.select_dtypes(float).columns] = df_rd.select_dtypes(float).round(3)
            st.dataframe(df_rd, use_container_width=True)

    if "preds" in eval_data:
        with st.expander("Sample Predictions (first 20 rows)"):
            st.dataframe(eval_data["preds"].head(20), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">ODIR-5K + RFMiD — Phase 2 Week 4 & Week 5</div>', unsafe_allow_html=True)
    odir_screenshots = sorted(ODIR_EVAL.glob("*.png")) if ODIR_EVAL.exists() else []
    if odir_screenshots:
        cols_sc = st.columns(len(odir_screenshots))
        for col_sc, sc_path in zip(cols_sc, odir_screenshots):
            with col_sc:
                st.image(str(sc_path), caption=sc_path.name, use_container_width=True)
    else:
        st.info("No screenshots found in eval_results/odir+rfmid/.")

    with st.expander("Raw Training Log — Phase 2 Week 4"):
        stats = load_stats()
        if stats:
            st.code(stats, language="text")

# ══════════════════════════════════════════════════════════════════════════════
# TAB: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("## 📂 Dataset Explorer")

    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([
        {"Dataset": "APTOS 2019",   "Images": "3,662",  "Task": "DR Severity (5 grades)",              "Used In": "P2 Wk3, P3 Wk6",        "Local Status": "✅ CSVs present; images on Kaggle"},
        {"Dataset": "ODIR-5K",      "Images": "7,000",  "Task": "Multi-disease (8 labels)",            "Used In": "P2 Wk4, P3 Wk5 & Wk6", "Local Status": "⚠️ Structure only; images on Kaggle"},
        {"Dataset": "RFMiD 2.0",    "Images": "3,200",  "Task": "Retinal disease classification",      "Used In": "P2 Wk4, P3 Wk5 & Wk6", "Local Status": "⚠️ Structure only; images on Kaggle"},
        {"Dataset": "Eye Diseases",  "Images": "~3,200", "Task": "4-class (Normal/DR/Cataract/Glaucoma)","Used In": "P3 Wk6",               "Local Status": "❌ Kaggle only"},
    ]), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">ODIR + RFMiD Class Distribution (N = 10,200)</div>', unsafe_allow_html=True)
    classes_dist = ["Normal", "DR", "Glaucoma", "Cataract", "AMD", "Other"]
    counts_dist  = [3191, 2888, 621, 340, 443, 2717]
    pcts_dist    = [31.3, 28.3, 6.1, 3.3, 4.3, 26.6]
    colors_dist  = ["#22c55e", "#3b82f6", "#f59e0b", "#ec4899", "#a855f7", "#64748b"]

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0f172a")
    for ax4 in (ax4a, ax4b):
        ax4.set_facecolor("#1e293b")
    dark_ax(ax4a)
    bars4 = ax4a.bar(classes_dist, counts_dist, color=colors_dist, alpha=0.85, edgecolor="#475569")
    for b4, pct in zip(bars4, pcts_dist):
        ax4a.text(b4.get_x() + b4.get_width()/2, b4.get_height() + 30, f"{pct}%",
                  ha="center", va="bottom", color="#e2e8f0", fontsize=9)
    ax4a.set_title("Class Counts", color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
    ax4a.set_ylabel("Count", color="#94a3b8")
    wedges, texts, autotexts = ax4b.pie(
        counts_dist, labels=classes_dist, colors=colors_dist,
        autopct="%1.1f%%", startangle=140,
        textprops={"color": "#e2e8f0", "fontsize": 9},
        wedgeprops={"edgecolor": "#0f172a", "linewidth": 2},
    )
    for at in autotexts:
        at.set_color("#0f172a"); at.set_fontweight("bold")
    ax4b.set_title("Class Distribution (%)", color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    st.markdown('<div class="section-header">APTOS 2019 — DR Grade Distribution</div>', unsafe_allow_html=True)
    aptos_csv = BASE_DIR / "data" / "general" / "APTOS2019" / "train_1.csv"
    if aptos_csv.exists():
        df_aptos = pd.read_csv(aptos_csv)
        grade_counts = df_aptos["diagnosis"].value_counts().sort_index()
        grade_name_map = {0: "Grade 0\nNo DR", 1: "Grade 1\nMild", 2: "Grade 2\nModerate", 3: "Grade 3\nSevere", 4: "Grade 4\nProliferative"}

        fig5, ax5 = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
        dark_ax(ax5)
        x5_labels = [grade_name_map.get(i, str(i)) for i in grade_counts.index]
        c5 = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]
        bars5 = ax5.bar(x5_labels, grade_counts.values, color=c5[:len(grade_counts)], alpha=0.85, edgecolor="#475569")
        for b5, val in zip(bars5, grade_counts.values):
            ax5.text(b5.get_x() + b5.get_width()/2, b5.get_height() + 10, str(val),
                     ha="center", va="bottom", color="#e2e8f0", fontsize=9, fontweight="bold")
        ax5.set_title(f"APTOS 2019 Grade Distribution (N={len(df_aptos):,})", color="#e2e8f0", fontsize=12, fontweight="bold", pad=10)
        ax5.set_ylabel("Count", color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**APTOS 2019 Grade Statistics**")
            # Build dataframe explicitly to avoid rename/assign column-name issues
            gc_df = pd.DataFrame({
                "Grade": grade_counts.index.map(grade_name_map),
                "Count": grade_counts.values,
            })
            gc_df["Percentage (%)"] = (gc_df["Count"] / gc_df["Count"].sum() * 100).round(1)
            st.dataframe(gc_df, hide_index=True, use_container_width=True)
        with col_s2:
            st.markdown("**Train / Val / Test Split (Phase 2 Week 4)**")
            st.dataframe(pd.DataFrame({
                "Split":      ["Train", "Validation", "Test"],
                "Count":      [7137, 1530, 1533],
                "Percentage": ["69.97%", "15.00%", "15.03%"],
            }), hide_index=True, use_container_width=True)
    else:
        st.info("APTOS 2019 CSV not found locally.")

    st.markdown('<div class="section-header">📥 Kaggle Download Requirements</div>', unsafe_allow_html=True)
    st.markdown("""
| Item | Kaggle Slug / Source | Required For | Size |
|---|---|---|---|
| APTOS 2019 images | `aptos2019-blindness-detection` | Phase 3 Wk6 training | ~3.4 GB |
| ODIR-5K images | `andrewmvd/ocular-disease-recognition-odir5k` | Phase 2 Wk4, Phase 3 Wk5+6 | ~4 GB |
| RFMiD 2.0 images | `andrewmvd/retinal-disease-classification` | Phase 2 Wk4, Phase 3 Wk5+6 | ~1.5 GB |
| Eye Diseases | `gunavenkatdoddi/eye-diseases-classification` | Phase 3 Wk6 training | ~500 MB |
| RETFound weights | HuggingFace `YukunZhou/RETFound_mae_natureCFP` | Phase 3 Wk6 (needs `HF_TOKEN`) | ~1.2 GB |
""")
    st.success("✅ **`models/aptos2019.pth`** (~270 MB) is already present locally — the Live Inference tab works without any downloads.")
