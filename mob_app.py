import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Internship Selection System", layout="centered")

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("Internship Selection Prediction System")

st.write("Bu sistem adayın staja seçilme olasılığını ve uygunluk skorunu tahmin eder.")

inputs = {}

for feature in feature_names:
    if feature == "CGPA":
        inputs[feature] = st.slider("CGPA", 0.0, 10.0, 7.0)
    elif feature in ["projects_count", "internships_done", "hackathons_participated", "certifications_count", "backlogs"]:
        inputs[feature] = st.slider(feature, 0, 10, 1)
    elif feature == "extracurricular":
        inputs[feature] = st.selectbox("extracurricular", [0, 1])
    else:
        inputs[feature] = st.slider(feature, 0, 100, 50)

input_df = pd.DataFrame([inputs], columns=feature_names)

threshold = st.slider("Firma seçim eşiği (%)", 0, 100, 50)

if st.button("Tahmin Et"):
    scaled_input = scaler.transform(input_df)

    probability = model.predict_proba(scaled_input)[0][1]
    score = probability * 100

    prediction = 1 if score >= threshold else 0

    st.subheader(f"Aday Uygunluk Skoru: %{score:.2f}")

    if prediction == 1:
        st.success("Sonuç: Aday seçilebilir.")
    else:
        st.error("Sonuç: Aday seçilmeyebilir.")

    st.write("### XAI - Özellik Önemleri")

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df)

    st.bar_chart(importance_df.set_index("Feature"))

    st.write("### SHAP Açıklaması")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    fig, ax = plt.subplots()

    try:
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0, :, 1],
                base_values=explainer.expected_value[1],
                data=input_df.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)

    except Exception as e:
        st.warning("SHAP grafiği gösterilemedi, ancak feature importance açıklaması yukarıda verilmiştir.")