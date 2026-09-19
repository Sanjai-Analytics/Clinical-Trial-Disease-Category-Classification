import pandas as pd
import numpy as np
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df=pd.read_csv(r'D:\sanjai\Tools\Visual Studio Code\Guvi Projects\Project 5\Dataset\clinical_trials_raw_patient2trial_conditions.csv')
print(pd.set_option('display.max_columns', None))
print(pd.set_option('display.width', None))

# print(df.head())
# print(df.info())
# print(df.describe().T)

# df=df.dropna(axis=0)
# print(df['phase'].notnull().sum())
# print(df['phase'].isnull().sum())

# print(df[df['source_condition_query']=='breast cancer', df['title']=='Prospective Cohort Study Depending on the Use of Palliative Care for Advanced Stage of Cancer Patients', df['conditions']=='Stage IV Breast Cancer | Stage IV Pancreatic Cancer | Stage IV Colon Cancer | Stage IV Gastric Cancer | Stage IV Lung Cancer | Stage IV Liver Cancer | Malignant Hematologic Neoplasm | Biliary Cancer Metastatic | Pediatric Leukemia | Pediatric Lymphoma | Pediatric Brain Tumor | Pediatric Solid Tumor', df['interventions']=='Early palliative care | Routine hospice care', df['overall_status']=='COMPLETED', df['study_type']=='OBSERVATIONAL', df['phase']].mean())

# mask = (
#     (df['source_condition_query'] == 'breast cancer') & 
#     (df['title'] == 'Prospective Cohort Study Depending on the Use of Palliative Care for Advanced Stage of Cancer Patients') & 
#     (df['conditions'] == 'Stage IV Breast Cancer | Stage IV Pancreatic Cancer | Stage IV Colon Cancer | Stage IV Gastric Cancer | Stage IV Lung Cancer | Stage IV Liver Cancer | Malignant Hematologic Neoplasm | Biliary Cancer Metastatic | Pediatric Leukemia | Pediatric Lymphoma | Pediatric Brain Tumor | Pediatric Solid Tumor') & 
#     (df['interventions'] == 'Early palliative care | Routine hospice care') & 
#     (df['overall_status'] == 'COMPLETED') & 
#     (df['study_type'] == 'OBSERVATIONAL')
# )

# print(df.loc[mask, 'phase'].value_counts().unique())



"""source_condition_query, title, offical_title, breif_summary, conditions, interventions, overall_status, study_type, phase, sex, healthy_volunteers, eligibility_criteria """

df.loc[(df['study_type']=='OBSERVATIONAL')&(df['phase'].isnull()), 'phase']='NA'
df['phase']=df['phase'].fillna('NA')
# print(df['phase'].value_counts(dropna=False))

# cols_to_fill = ['official_title', 'interventions', 'conditions']
# df[cols_to_fill] = df[cols_to_fill].fillna('Not Specified')

df['sex'] = df['sex'].fillna('ALL')

df['healthy_volunteers'] = df['healthy_volunteers'].fillna(False)

df=df.drop(columns=['maximum_age', 'minimum_age', 'clinicaltrials_url', 'nct_id'])
df=df.dropna(axis=0)

# print(df.info())
# print(df.head())

text_cols=['title', 'official_title', 'brief_summary', 'conditions', 'interventions', 'eligibility_criteria']

for col in text_cols:
    df[col]=df[col].apply(lambda x: unidecode(str(x)) if pd.notnull(x) and x != '' else x)
    df[col]=df[col].str.replace(r'\n|\r',' ', regex=True)
    df[col]=df[col].str.replace('%', ' percent ', regex=False)
    df[col]=df[col].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    df[col]=df[col].str.lower()
    df[col]=df[col].str.replace(r'\s+', ' ', regex=True).str.strip()

df['master_text']=(
    df['title'] + " " +
    df['brief_summary'] + " " +
    df['conditions'] + " " +
    df['interventions'] + " " +
    df['eligibility_criteria']
)



tfid=TfidfVectorizer(stop_words='english', max_features=5000)
X_test=tfid.fit_transform(df['master_text'])
# print("Shape of TF-IDE Matrix:", X_test.shape)

cat_cols = ['overall_status', 'study_type', 'phase', 'sex']
column_transformer=ColumnTransformer(
    transformers=[
        ('One_hot_encoder', OneHotEncoder(sparse_output=True), cat_cols)   
    ],
    remainder='drop'
)

x_categorical=column_transformer.fit_transform(df)

x_categorical_scaled=x_categorical*0.1
X_final=hstack([X_test,x_categorical_scaled])
# X_final=hstack([X_test,x_categorical])
# print("Final matrix shape:",X_final.shape)
# print(X_final)

inertia=[]
K_range=range(2,10)

for k in K_range:
    kmeans=KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_final)
    inertia.append(kmeans.inertia_)
    # print(f"Tested k={k}, Inertia: {kmeans.inertia_}")


# plt.figure(figsize=(10,6))
# plt.plot(K_range, inertia, marker='o', linestyle='--')
# plt.title('Elbow Method For Optimal k (Clinical Trial)')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Inertia (Sum of Squared Errors)')
# plt.xticks(K_range)
# plt.grid(True)
# plt.show()


# final_kmeans=KMeans(n_clusters=5, random_state=42, n_init='auto')

# cluster_labels=final_kmeans.fit_predict(X_final)

# df['Cluster']=cluster_labels

# print(df['Cluster'].value_counts())

# print(df[df['Cluster']==0][['title', 'study_type', 'Cluster']].head())

# sil_score=silhouette_score(X_final, cluster_labels, sample_size=10000, random_state=42)

# print(f"Silhouette score for k=5: {sil_score:.4f}")

# sil_score=[]
# K_range=range(2,11)
# for k in K_range:
#     kmeans=KMeans(n_clusters=k, random_state=42, n_init='auto')
#     labels=kmeans.fit_predict(X_final)
#     score=silhouette_score(X_final, labels, sample_size=10000, random_state=42)
#     sil_score.append(score)
#     print(f"Tested k={k}, Silhouettle Score: {score:.4f}")

# plt.figure(figsize=(10,6))
# plt.plot(K_range, sil_score, marker='s', color='purple', linestyle='-')
# plt.title('Silhouette Score vs. Number of Clusters (k)')
# plt.xlabel('Number of Clusters (k)')
# plt.ylabel('Silhouette Score (Higher is Better)')
# plt.xticks(K_range)
# plt.grid(True)
# plt.show()



final_k = 6
final_kmeans = KMeans(n_clusters=final_k, random_state=42, n_init='auto')
df['Cluster'] = final_kmeans.fit_predict(X_final)

# text_features = tfid.get_feature_names_out()
# cat_features = column_transformer.get_feature_names_out()

# all_features = np.concatenate([text_features, cat_features])

# # print(f"TOP 10 FEATURES PER CLUSTER (k={final_k})")
# for i in range(final_k):
#     center = final_kmeans.cluster_centers_[i]
    
#     top_10_indices = center.argsort()[::-1][:10]
#     top_features = all_features[top_10_indices]
    
    # print(f"\nCluster {i}:")
    # print(", ".join(top_features))


cluster_mapping = {
    0: "COVID-19 & Infectious Diseases",
    1: "Diabetes & Endocrinology",
    2: "Breast Cancer & Oncology",
    3: "Ophthalmology",
    4: "Mental Health & Pain Management",
    5: "Chronic & Inflammatory Diseases"
}

df['Disease_category']=df['Cluster'].map(cluster_mapping)
# # df.to_csv('Clustered_clinical_trials.csv', index=False)

df=df.drop(columns=['Cluster'])
x=X_test
y=df['Disease_category']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
classifier=LogisticRegression(max_iter=1000)
classifier.fit(x_train, y_train)

y_hat=classifier.predict(x_test)
accuracy=accuracy_score(y_test,y_hat)
# print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

# print("\nClassification Report:")
# print(classification_report(y_test, y_hat))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# # 1. Generate the confusion matrix
# cm = confusion_matrix(y_test, y_hat, labels=classifier.classes_)

# # 2. Plot it using a Seaborn heatmap
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=classifier.classes_, 
#             yticklabels=classifier.classes_)

# plt.title('Disease Classification Confusion Matrix', fontsize=16)
# plt.ylabel('Actual True Disease', fontsize=12)
# plt.xlabel('Machine Learning Prediction', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()


# joblib.dump(tfid, 'tfidf_vectorizer.pkl')

# joblib.dump(classifier, 'disease_classifier_model.pkl')




