import streamlit as st
import joblib
import pandas as pd
import re
from unidecode import unidecode
from PIL import Image
import matplotlib.pyplot as plt


st.set_page_config(page_title="Clinical Trial Predictor", layout="centered")

@st.cache_resource
def load_models():
    tfid = joblib.load('tfidf_vectorizer.pkl')
    classifier = joblib.load('disease_classifier_model.pkl')
    return tfid, classifier

tfid, classifier = load_models()

def clean_text(text):
    if pd.isnull(text) or text == '':
        return ""
    text = unidecode(str(text))
    text = re.sub(r'\n|\r', ' ', text)
    text = text.replace('%', ' percent ')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

st.title("Clinical Trial Disease Classifier")

tab1, tab2 = st.tabs(["Disease Predictor", "Model Analytics"])

with tab1:
    st.write("""
    Enter the **Brief Summary** of a clinical trial below to predict its disease category.
    """)

    user_input = st.text_area("Enter Clinical Trial Text Here:", height=200, 
                              placeholder="e.g., This study evaluates the efficacy of chemotherapy...")

    if st.button("Predict Disease Category"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner('Analyzing medical text...'):
                cleaned_input = clean_text(user_input)
                vectorized_input = tfid.transform([cleaned_input])
                prediction = classifier.predict(vectorized_input)[0]
                
                st.success("Prediction Complete!")
                st.metric(label="Predicted Category", value=prediction)


with tab2:
    st.header("Machine Learning Analytics")
    st.write("This model was built in two phases: Unsupervised Clustering to discover disease categories and Supervised Classification to predict them.")
    st.divider()

    st.subheader("1. Classification Accuracy")
    st.write("Logistic Regression model achieved a **98.47% Accuracy**. Below is the Confusion Matrix showing where the model succeeded and the minor areas of overlapping medical vocabulary.")
    
    try:
        img_cm = Image.open(r"D:\sanjai\studies\institution\Guvi\Projects by Guvi\Project 5\Dataset\Charts\Disease Classification Confusion Matrix.png")
        
        st.image(img_cm, caption="Confusion Matrix: Actual vs. Predicted Diseases")
    except FileNotFoundError:
        st.error("Image 'Disease Classification Confusion Matrix.png' not found.")

    st.divider()

    
    st.subheader("2. Discovering the Clusters")
    st.write("**K-Means Clustering** to mathematically discover 6 distinct medical themes in the raw text data.")   
     
    
    col_elbow, col_sil = st.columns(2)
    
    with col_elbow:
        try:
            img_elbow = Image.open(r"D:\sanjai\studies\institution\Guvi\Projects by Guvi\Project 5\Dataset\Charts\2nd Time Eblow Method for optimal k (Clinical Trial).png")
            st.image(img_elbow, caption="Elbow Method showing diminishing returns")
        except FileNotFoundError:
            st.error("Elbow Method image not found.")

    with col_sil:
        try:
            img_sil = Image.open(r"D:\sanjai\studies\institution\Guvi\Projects by Guvi\Project 5\Dataset\Charts\2nd Time Silhouette score VS Numer of clusters.png")
            st.image(img_sil, caption="Silhouette Score peaking at k=6")
        except FileNotFoundError:
            st.error("Silhouette Score image not found.")
    
    st.divider()

    
    st.subheader("3. Disease Category Distribution")
    st.write("The unsupervised clustering segmented the dataset into six distinct, imbalanced medical categories:")
    
    Disease_Categories = ['Oncology & Breast Cancer','Mental Health & Pain Management','Diabetes & Endocrinology', 'Chronic & Inflammatory Diseases', 'COVID-19 & Infectious Diseases', 'Ophthalmology']
    Disease_Percentage = [25.8, 23.6, 18.2, 15.7, 13.3, 3.1]
    
    fig = plt.figure(figsize=(10, 6))
    plt.pie(
        x=Disease_Percentage,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.1,0,0,0,0,0),
        shadow=True,
        pctdistance=0.75,
        radius=1.25    
    )
    plt.title("Disease Category Distribution", fontweight='bold')
    plt.legend(Disease_Categories,
               title="Categories",
               loc="center left",
               bbox_to_anchor=(1.2, 0.5), 
               fontsize=10)
    
    st.pyplot(fig, use_container_width=False)

    st.divider()

    
    st.subheader("4. Frequently Occurring Medical Terms")
    st.write("The TF-IDF vectorization revealed that clinical trials can be reliably classified using specific, high-frequency biological and treatment markers:")

    keyword_data = {
        "Disease Category": ["Infectious Disease", "Endocrinology", "Oncology", "Ophthalmology", "Psychiatry", "Inflammatory"],
        "Top Identifying Keywords": ["covid, sars, cov, infection", "insulin, glucose, mellitus", "metastatic, chemotherapy, tumor", "glaucoma, ocular, iop", "anxiety, disorder, pain", "copd, pulmonary, rheumatoid"]
    }
    df_keywords = pd.DataFrame(keyword_data).set_index("Disease Category")
    st.table(df_keywords)

    st.subheader("5. Clinical Trial Text Patterns & Trends")
    st.write("Exploratory data analysis revealed several critical patterns in medical text:")

    st.info("**Administrative vs. Clinical Language:** Raw clinical summaries are heavily saturated with administrative terms (e.g., 'completed', 'interventional'). Without scaling, models group trials by administrative status rather than medical disease.")
    st.warning("**Overlapping Post-Operative Vocabulary:** The model identified a trend where severe disease trials (like Breast Cancer surgery) share significant vocabulary with Pain Management trials (e.g., 'preventive therapy', 'complications').")
    st.success("**The Power of Demographics:** Categorical constraints (such as `sex_FEMALE`) act as powerful anchors for text clustering, helping the model mathematically isolate subsets like women's health.")

    st.subheader("6. Model Performance")
    st.write("The NLP and machine learning pipeline demonstrated enterprise-grade performance:")

    col_acc, col_f1, col_err = st.columns(3)
    col_acc.metric(label="Overall Accuracy", value="98.47%")
    col_f1.metric(label="F1-Score Range", value="0.97 - 0.99")
    col_err.metric(label="Error Rate", value="< 1.5%")

    # st.caption("**Error Analysis:** The Confusion Matrix visualization showed that the rare misclassifications occurred primarily when critical disease keywords (like 'cancer') were omitted from the text, causing the algorithm to default to secondary topics.")

    st.divider()
    st.subheader("7. Supporting Healthcare Analytics & Decision-Making")
    st.write("This end-to-end pipeline proves that unstructured medical text can be automated and categorized with high reliability. For a healthcare organization, this tool:")

    col_biz1, col_biz2, col_biz3 = st.columns(3)

    with col_biz1:
        st.markdown("### Reduces Manual Review")
        st.write("Automates the sorting of thousands of unstructured clinical trial protocols, saving massive amounts of manual medical review time.")

    with col_biz2:
        st.markdown("### Enhances Searchability")
        st.write("Allows researchers and AI-driven management systems to instantly query and retrieve historical trials based on distinct medical themes.")

    with col_biz3:
        st.markdown("### Enables Rapid Triage")
        st.write("New clinical trial summaries can be instantly diagnosed and routed to the correct medical department using this exact web application.")