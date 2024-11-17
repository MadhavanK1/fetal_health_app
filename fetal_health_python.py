import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

st.title('Fetal Health Classification: A Machine Learning App')
st.image('fetal_health_image.gif')
st.write("Utilize our advanced machine learning application to predict fetal health classification.")

st.sidebar.header('Fetal Health Features Input')

uploaded_file = st.sidebar.file_uploader("Upload your data", type=['csv'])

st.sidebar.warning('⚠️ Ensure your data strictly follows the format outlined below.')

st.sidebar.write("### Sample Data Format")
sample_data = pd.read_csv('fetal_health.csv').head()
st.sidebar.dataframe(sample_data)

model_choice = st.sidebar.radio(
    '**Choose Model for Prediction**',
    ('Random Forest', 'Decision Tree', 'AdaBoost', 'Soft Voting')
)

st.sidebar.info(f"**You selected: {model_choice}**") 

if uploaded_file is None:
    st.info('📤 ***Please upload data to proceed.***')
else:
    st.success('✅ ***CSV file uploaded successfully.***')

rf_pickle = open('random_forest_model.pickle', 'rb')
rf_model = pickle.load(rf_pickle)
rf_pickle.close()

dt_pickle = open('decision_tree_model.pickle', 'rb')
dt_model = pickle.load(dt_pickle)
dt_pickle.close()

ada_pickle = open('adaboost_model.pickle', 'rb')
ada_model = pickle.load(ada_pickle)
ada_pickle.close()

voting_pickle = open('voting_classifier_model.pickle', 'rb')
voting_clf = pickle.load(voting_pickle)
voting_pickle.close()

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    
    if model_choice == 'Random Forest':
        selected_model = rf_model
    elif model_choice == 'Decision Tree':
        selected_model = dt_model
    elif model_choice == 'AdaBoost':
        selected_model = ada_model
    elif model_choice == 'Soft Voting':
        selected_model = voting_clf
        
    predictions = selected_model.predict(input_df)
    probabilities = selected_model.predict_proba(input_df)

    label_map = {
        1: 'Normal',
        2: 'Suspect',
        3: 'Pathological'
    }
    text_predictions = []
    for pred in predictions:
        text_label = label_map[pred]
        text_predictions.append(text_label)
    
    results_df = input_df.copy()
    results_df['Predicted Fetal Health'] = text_predictions
    results_df['Prediction Probability (%)'] = np.max(probabilities, axis=1)*100
    
    def get_color(value):
        colors = {
            'Normal': 'background-color: lime',
            'Suspect': 'background-color: yellow',
            'Pathological': 'background-color: orange'
        }
        return colors.get(value, '')
    
    styled_df = results_df.style.applymap(get_color, subset=['Predicted Fetal Health']).format({'Prediction Probability (%)': '{:.2f}'})
    
    st.write(f"### Predicting Fetal Health Class Using {model_choice} Model")
    st.dataframe(styled_df)

st.subheader("Model Performance and Insights")
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

with tab1:
    st.write("### Confusion Matrix")
    if model_choice == 'Random Forest':
        st.image('confusion_mat_random_forest.svg')
    elif model_choice == 'Decision Tree':
        st.image('confusion_mat_decision_tree.svg')
    elif model_choice == 'AdaBoost':
        st.image('confusion_mat_adaboost.svg')
    else:  
        st.image('confusion_mat_voting_classifier.svg')
    st.caption("Confusion Matrix of model predictions.")

with tab2:
    st.write("### Classification Report")
    if model_choice == 'Random Forest':
        report_df = pd.read_csv('class_report_random_forest.csv', index_col = 0)
    elif model_choice == 'Decision Tree':
        report_df = pd.read_csv('class_report_decision_tree.csv', index_col = 0)
    elif model_choice == 'AdaBoost':
        report_df = pd.read_csv('class_report_adaboost.csv', index_col = 0)
    else: 
        report_df = pd.read_csv('class_report_voting_classifier.csv', index_col = 0)
    st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
    st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each class.")

with tab3:
    st.write("### Feature Importance")
    if model_choice == 'Random Forest':
        st.image('feature_imp_random_forest.svg')
    elif model_choice == 'Decision Tree':
        st.image('feature_imp_decision_tree.svg')
    elif model_choice == 'AdaBoost':
        st.image('feature_imp_adaboost.svg')
    else:  
        st.image('feature_imp_voting_classifier.svg')
    st.caption("Features used in this prediction are ranked by relative importance.")

    

