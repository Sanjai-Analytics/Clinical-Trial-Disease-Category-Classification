# Clinical-Trial-Disease-Category-Classification
Here is a professional, portfolio-ready README.md file for your GitHub repository. It perfectly summarizes your technical pipeline, business impact, and instructions for how recruiters or evaluators can run your Streamlit app.

You can copy and paste this directly into the README.md file in your GitHub repo.

Clinical Trial Disease Classification System
Project Overview
The healthcare industry generates massive volumes of unstructured text data through clinical trial summaries. Sorting and categorizing these trials historically requires thousands of hours of manual human review.

This project is an end-to-end Natural Language Processing (NLP) and Machine Learning pipeline designed to automatically read, process, and classify unstructured clinical trial summaries into specific disease categories. The project features a custom-built interactive web dashboard for real-time medical triage.

Key Features
Unsupervised Text Clustering: Utilized K-Means clustering (optimized via Elbow Method and Silhouette Scores) to mathematically discover 6 distinct medical themes from raw text data.

NLP Text Processing: Implemented robust text cleaning and TF-IDF (Term Frequency-Inverse Document Frequency) vectorization, specifically filtering out domain-specific "medical stopwords" (e.g., patient, trial, dose) to isolate high-value biological keywords.

Supervised Classification: Trained a Logistic Regression classifier achieving 98.47% accuracy in predicting unseen clinical trials.

Interactive Dashboard: Deployed a fully functional Streamlit application allowing users to input raw trial summaries and receive instant disease category predictions alongside model analytics.

Disease Categories Identified
Oncology & Breast Cancer (25.8%)

Mental Health & Pain Management (23.6%)

Diabetes & Endocrinology (18.2%)

Chronic & Inflammatory Diseases (15.7%)

COVID-19 & Infectious Diseases (13.3%)

Ophthalmology (3.1%)

Technology Stack
Language: Python

Machine Learning: Scikit-Learn (TfidfVectorizer, KMeans, LogisticRegression)

Data Manipulation: Pandas, NumPy, Regex, Unidecode

Data Visualization: Matplotlib, Seaborn, PIL

Frontend/Deployment: Streamlit
