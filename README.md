# 📚 Student Job & Course Recommendation System

A data science project that helps students discover personalized online courses and job opportunities based on their academic background, skills, and career goals. This system uses machine learning models and a user-friendly Streamlit interface to make real-time predictions.

---

## 🚀 Project Overview

Students often struggle to identify the right career path or relevant learning resources. Our system bridges that gap using:

- Student profile analysis (academic data + skillset)
- Machine learning for course category prediction
- Cosine similarity for job-role matching
- Real-time recommendations via an interactive dashboard

---

## 🛠️ Tech Stack

**Languages & Libraries**
- Python
- Pandas, NumPy
- Scikit-learn (ML models & evaluation)
- Matplotlib, Seaborn (visualization)
- Streamlit (dashboard)

**Tools**
- Visual Studio Code
- Joblib (model serialization)

---

## 📊 Datasets Used

All datasets are sourced from public Kaggle repositories:

1. **Online Courses Dataset**  
   https://www.kaggle.com/datasets/khaledatef1/online-courses

2. **Job Postings Dataset**  
   https://www.kaggle.com/datasets/filibertozurita/job-postings

3. **Student Skillset Dataset**  
   https://www.kaggle.com/datasets/kushagrathisside/student-skillset-analysis

---

## 💡 Key Features

- **TF-IDF + PCA** for transforming and compressing resume data
- **Label Encoding** for categorical variables like branch and course type
- **Decision Tree Classifier** for course prediction
- **Cosine Similarity** to recommend relevant jobs
- **SHAP** for interpreting model predictions
- **Streamlit App** with tabs for:
  - Home
  - Predictions
  - Data Analysis
  - About

---

##  🎥 Platform Demo Video

- **Link to the Video: https://drive.google.com/file/d/12zQ7RBTS3DZrD9i1JlJw7-Flm1RrflIz/view?usp=sharing**

---

## 🖥️ How to Run Locally

1. Clone the repository:
   git clone https://github.com/SABO007/cap5771sp25-project.git

2. Make sure the following folders exist:
```bash
├── Data/
│   └── All the generated datasets
│
├── Models/
│   ├── best_model.pkl
│   ├── job_tfidf_matrix.pkl
│   ├── job_tfidf_vectorizer.pkl
│   ├── job_titles.pkl
│   ├── label_encoders.pkl
│   ├── pca_vectorizer.pkl
│   └── tfidf_vectorizer.pkl
│
├── Report/
│   ├── MILESTONE1.pdf
│   └── MILESTONE2.pdf
│   └── Final Report.pdf
│
├── Scripts/
│   ├── Data_merging.ipynb     # Merging the datsets
│   ├── EDA.ipynb              # Exploratory data analysis
│   ├── model_evaluation.ipynb # Post model evaluation Feature importance and SHAP
│   ├── Modelling.ipynb        # Feature Engineering, Evaluation, Model Training 
│   └── Preprocess.ipynb       # Data Preprocessing
│
├── app.py                     # Streamlit Dashboard file
└── README.md

```

3. Launch the app:
```bash
   streamlit run app.py
```

---

## 👥 Team Contributions

- **Sashank Boppana**: Data processing, model training, evaluation, SHAP interpretation  
- **Tejesh Boppana**: Streamlit UI development, dashboard integration with backend, frontend interaction

---

## 📌 License

This project is developed for academic purposes. Refer to individual datasets on Kaggle for their specific licenses.

---

## 🙌 Acknowledgements

Thanks to Kaggle for providing accessible datasets and to the open-source community for tools like Streamlit and Scikit-learn.
