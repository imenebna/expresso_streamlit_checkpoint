# I import what I need to build my Streamlit app, handle tabular inputs, and load my trained model.
import streamlit as st
import pandas as pd
import joblib

# I set the page configuration so my app looks clean and consistent from the first second.
st.set_page_config(
    page_title="Expresso Churn - Checkpoint",
    page_icon="📶",
    layout="centered",
)

# I present the goal of my app in a simple and readable way.
st.title("📶 Expresso Churn Prediction")
st.caption("Enter client features, click Predict, and get a churn probability with a corresponding risk.")

st.divider()

# I load my trained pipeline once to keep the app fast during user interactions.
@st.cache_resource
def load_model():
    return joblib.load("expresso_churn_model.joblib")

model = load_model()

# I load my dataset (if available) to offer dropdown choices and reduce typing mistakes.
@st.cache_data
def load_dataset_for_choices(path: str = "Expresso_churn_dataset.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

df_choices = load_dataset_for_choices()

# I extract clean dropdown options from the dataset when it exists.
def build_choices(df: pd.DataFrame | None):
    choices = {}
    if df is None:
        return choices

    cat_cols = ["REGION", "TENURE", "TOP_PACK", "MRG"]
    for c in cat_cols:
        if c in df.columns:
            vals = (
                df[c]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .unique()
                .tolist()
            )
            vals = sorted(set(vals))
            choices[c] = vals
    return choices

choices = build_choices(df_choices)

# I translate the predicted probability into a simple business-friendly risk label.
def risk_label(p: float):
    if p < 0.33:
        return "Low risk"
    if p < 0.66:
        return "Medium risk"
    return "High risk"

# I keep a single source of truth for the exact columns my model can require at prediction time.
REQUIRED_COLS = [
    "REGION", "TENURE", "TOP_PACK", "MRG",
    "MONTANT", "FREQUENCE_RECH", "REVENUE", "ARPU_SEGMENT", "FREQUENCE",
    "DATA_VOLUME", "ON_NET", "ORANGE", "TIGO",
    "ZONE1", "ZONE2", "REGULARITY", "FREQ_TOP_PACK",
]

TEXT_COLS = ["REGION", "TENURE", "TOP_PACK", "MRG"]

# I build the UI inside a form so users fill everything first, then predict in one click.
with st.form("predict_form"):
    st.subheader("1) Client information")

    col1, col2, col3, col4 = st.columns(4)

    # I let users pick from known categories (when available) to prevent out-of-range values.
    with col1:
        if choices.get("REGION"):
            region = st.selectbox("REGION", ["Select..."] + choices["REGION"])
        else:
            region = st.text_input("REGION", placeholder="e.g., DAKAR")

    with col2:
        if choices.get("TENURE"):
            tenure = st.selectbox("TENURE", ["Select..."] + choices["TENURE"])
        else:
            tenure = st.text_input("TENURE", placeholder="e.g., K >12 month")

    with col3:
        if choices.get("TOP_PACK"):
            top_pack = st.selectbox("TOP_PACK", ["Select..."] + choices["TOP_PACK"])
        else:
            top_pack = st.text_input("TOP_PACK", placeholder="e.g., MIX")

    with col4:
        if choices.get("MRG"):
            mrg = st.selectbox("MRG", ["Select..."] + choices["MRG"])
        else:
            mrg = st.text_input("MRG", placeholder="e.g., YES/NO (as in your dataset)")

    st.subheader("2) Usage & billing variables")

    c1, c2 = st.columns(2)

    # I collect numeric inputs with safe defaults so the app never crashes on empty values.
    with c1:
        montant = st.number_input("MONTANT", min_value=0.0, value=0.0, step=1.0)
        frequence_rech = st.number_input("FREQUENCE_RECH", min_value=0.0, value=0.0, step=1.0)
        revenue = st.number_input("REVENUE", min_value=0.0, value=0.0, step=1.0)
        arpu_segment = st.number_input("ARPU_SEGMENT", min_value=0.0, value=0.0, step=1.0)
        frequence = st.number_input("FREQUENCE", min_value=0.0, value=0.0, step=1.0)

    with c2:
        data_volume = st.number_input("DATA_VOLUME", min_value=0.0, value=0.0, step=1.0)
        on_net = st.number_input("ON_NET", min_value=0.0, value=0.0, step=1.0)
        orange = st.number_input("ORANGE", min_value=0.0, value=0.0, step=1.0)
        tigo = st.number_input("TIGO", min_value=0.0, value=0.0, step=1.0)
        regularity = st.number_input("REGULARITY", min_value=0.0, value=0.0, step=1.0)

    st.subheader("3) Zones")

    z1, z2, z3 = st.columns(3)
    with z1:
        zone1 = st.number_input("ZONE1", min_value=0.0, value=0.0, step=1.0)
    with z2:
        zone2 = st.number_input("ZONE2", min_value=0.0, value=0.0, step=1.0)
    with z3:
        freq_top_pack = st.number_input("FREQ_TOP_PACK", min_value=0.0, value=0.0, step=1.0)

    submitted = st.form_submit_button("✅ Predict churn probability")

# I validate inputs, build the final DataFrame with the exact schema, and run the model prediction.
if submitted:
    # I make sure no key categorical field is left empty or stuck on "Select...".
    for name, val in [("REGION", region), ("TENURE", tenure), ("TOP_PACK", top_pack), ("MRG", mrg)]:
        if str(val).strip() == "" or val == "Select...":
            st.error(f"Please provide a valid value for {name}.")
            st.stop()

    # I build a one-row DataFrame and I explicitly include MRG so the pipeline never complains.
    X_input = pd.DataFrame([{
        "REGION": region,
        "TENURE": tenure,
        "TOP_PACK": top_pack,
        "MRG": mrg,
        "MONTANT": montant,
        "FREQUENCE_RECH": frequence_rech,
        "REVENUE": revenue,
        "ARPU_SEGMENT": arpu_segment,
        "FREQUENCE": frequence,
        "DATA_VOLUME": data_volume,
        "ON_NET": on_net,
        "ORANGE": orange,
        "TIGO": tigo,
        "ZONE1": zone1,
        "ZONE2": zone2,
        "REGULARITY": regularity,
        "FREQ_TOP_PACK": freq_top_pack,
    }])

    # I guarantee the exact final column order and I add any missing required column defensively.
    for c in REQUIRED_COLS:
        if c not in X_input.columns:
            X_input[c] = pd.NA
    X_input = X_input[REQUIRED_COLS]

    # I standardize missing values so preprocessing behaves predictably.
    for c in TEXT_COLS:
        X_input[c] = (
            X_input[c]
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "<NA>": "Unknown"})
            .fillna("Unknown")
        )

    num_cols = [c for c in REQUIRED_COLS if c not in TEXT_COLS]
    for c in num_cols:
        X_input[c] = pd.to_numeric(X_input[c], errors="coerce").fillna(0)

    # I compute churn probability and convert it into an interpretable risk level.
    proba = float(model.predict_proba(X_input)[0][1])
    label = risk_label(proba)

    st.success("Prediction completed.")
    st.metric("Churn probability", f"{proba:.1%}", help="Probability the client will churn (class 1).")

    # I communicate the result
    if label == "Low risk":
        st.success("Risk level: Low risk. This client looks relatively stable according to my model.")
    elif label == "Medium risk":
        st.warning("Risk level: Medium risk. I would monitor this client and consider light retention action.")
    else:
        st.error("Risk level: High risk. I would recommend strong retention action for this client.")

    # I show exactly what was sent to the model
    with st.expander("Show the input data used for prediction"):
        st.dataframe(X_input, width="stretch")
