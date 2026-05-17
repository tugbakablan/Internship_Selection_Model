import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Internship Selection Prediction",
    layout="centered"
)

st.title("Internship Selection Prediction System")

st.write(
    """
Bu sistem öğrencinin staja seçilme olasılığını tahmin eder,
uygunluk skorunu hesaplar ve XAI açıklamaları sunar.
"""
)

model = joblib.load("rf_model.joblib")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.header("Aday Bilgileri")

inputs = {}

for feature in feature_names:

    if feature == "CGPA":

        inputs[feature] = st.slider(
            "CGPA",
            0.0,
            10.0,
            7.0,
            0.1
        )

    elif feature in [
        "projects_count",
        "internships_done",
        "hackathons_participated",
        "certifications_count",
        "backlogs"
    ]:

        inputs[feature] = st.slider(
            feature,
            0,
            10,
            1
        )

    elif feature == "extracurricular":

        inputs[feature] = st.selectbox(
            "extracurricular",
            [0, 1]
        )

    else:

        inputs[feature] = st.slider(
            feature,
            0,
            100,
            50
        )


st.header("Şirket Seçim Eşiği")

threshold = st.slider(
    "Selection Threshold (%)",
    0,
    100,
    50
)


input_df = pd.DataFrame(
    [inputs],
    columns=feature_names
)


if st.button("Değerlendir"):

    scaled_input = scaler.transform(input_df)

    probability = model.predict_proba(
        scaled_input
    )[0][1]

    score = probability * 100

    prediction = 1 if score >= threshold else 0


    st.header("Sonuç")

    st.subheader(
        f"Aday Uygunluk Skoru: %{score:.2f}"
    )

    if prediction == 1:

        st.success(
            "Sonuç: Aday seçilebilir."
        )

    else:

        st.error(
            "Sonuç: Aday seçilmeyebilir."
        )


    st.write("### Sistem Yorumu")

    if score >= 80:

        st.write(
            "Aday güçlü bir profil göstermektedir."
        )

    elif score >= 60:

        st.write(
            "Aday orta-üst seviyede uygun görünmektedir."
        )

    else:

        st.write(
            "Adayın geliştirilmesi gereken alanları bulunmaktadır."
        )


    st.write("## XAI | Feature Importance")

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )


    st.write("### En Etkili Özellikler")

    top_features = importance_df.head(3)

    for _, row in top_features.iterrows():

        st.write(
            f"- {row['Feature']} "
            f"(Importance: {row['Importance']:.3f})"
        )


    st.write("## XAI Yorumu")

    try:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(
            scaled_input
        )

        explanation = shap.Explanation(

            values=shap_values[0, :, 1],

            base_values=explainer.expected_value[1],

            data=input_df.iloc[0],

            feature_names=feature_names
        )

        fig, ax = plt.subplots(figsize=(10,5))

        shap.plots.waterfall(
            explanation,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning(
            "SHAP grafiği gösterilemedi."
        )

        st.text(str(e))