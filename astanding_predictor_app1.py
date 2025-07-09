
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Title
st.title("ASTANDING Prediction Web App")
st.write("Upload an Excel file with Sheet6 (training) and Sheet7 (prediction) to get academic standing predictions.")

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Load training data
    df = pd.read_excel(uploaded_file, sheet_name="Sheet6")

    ASTANDINGEN = {
        'Academic Probation 1 Term': 1,
        'Academic Probation 2 Terms': 2,
        'Student earned 75% or more': 3,
        'Student earned around 75%': 3,
		'Repeating Year': 3,
        'Resit Exam': 3,
        'Remedial Term Required': 3,
        'Dismissed & Major Changed': 4,		
        'Dismissed After Remedial Term': 4,
        'Dismissed with No Rem. Term': 4,
        'Dismissed': 4,
        'Continue Normally After Remed.': 5,
        'Resit and ChangedtoNormal Memo': 5,
		'Good Standing': 5
    }
    ASTANDINGDECODE = {v: k for k, v in ASTANDINGEN.items()}

    df['ASTANDING_ENCODED'] = df['ASTANDING'].map(ASTANDINGEN)
    df.dropna(subset=['ASTANDING_ENCODED'], inplace=True)


    X = df.drop(columns=['ASTANDING', 'ASTANDING_ENCODED'])
    y = df['ASTANDING_ENCODED']
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train_resampled, y_train_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
    model.fit(X_train_resampled, y_train_resampled)

    # Predict from Sheet7
    try:
        new_df = pd.read_excel(uploaded_file, sheet_name="Sheet7")


        for col in X.columns:
            if col not in new_df.columns:
                new_df[col] = 0
        new_df = new_df[X.columns]

        predictions = model.predict(new_df)
        new_df['Predicted_ASTANDING'] = [ASTANDINGDECODE[p] for p in predictions]

        st.success("Predictions completed. Preview below:")
        st.dataframe(new_df[['Predicted_ASTANDING']])

        # Download option
        output_file = "predicted_astanding.xlsx"
        new_df.to_excel(output_file, index=False)
        with open(output_file, "rb") as f:
            st.download_button("Download Predictions as Excel", f, file_name=output_file)

    except Exception as e:
        st.error(f"Error reading Sheet7: {e}")
