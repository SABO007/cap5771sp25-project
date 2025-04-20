import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Course & Job Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Main title
st.title("📚 Course Category & Job Recommendation System")
st.write("This application helps predict course categories based on student profile and skills, and suggests potential job opportunities.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predictions", "Data Analysis", "About"])

# Function to load models and data
@st.cache_resource
def load_models():
    try:
        model = joblib.load('Models/best_model.pkl')
        pca = joblib.load('Models/pca_vectorizer.pkl')
        tfidf = joblib.load('Models/tfidf_vectorizer.pkl')
        label_encoders = joblib.load('Models/label_encoders.pkl')
        job_vectorizer = joblib.load('Models/job_tfidf_vectorizer.pkl')
        job_tfidf_matrix = joblib.load('Models/job_tfidf_matrix.pkl')
        job_titles = joblib.load('Models/job_titles.pkl')
        
        return {
            'model': model,
            'pca': pca,
            'tfidf': tfidf,
            'label_encoders': label_encoders,
            'job_vectorizer': job_vectorizer,
            'job_tfidf_matrix': job_tfidf_matrix,
            'job_titles': job_titles
        }
    except FileNotFoundError:
        st.error("Model files not found. Please make sure the models are in the correct directory.")
        return None

@st.cache_data
def load_data():
    try:
        student_course_data = pd.read_csv('Data/final_student_course.csv')
        job_data = pd.read_csv('Data/final_job_data.csv')
        return student_course_data, job_data
    except FileNotFoundError:
        st.error("Data files not found. Please make sure the data files are in the correct directory.")
        return None, None

# Home page
if page == "Home":
    st.header("Welcome to Course & Job Recommendation System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About this App")
        st.write("""
        This application uses machine learning to help students:
        - Predict suitable course categories based on their profile and skills
        - Find potential job opportunities matching their interests
        - Explore data insights about courses and job market trends
        
        Navigate using the sidebar to explore different features!
        """)
    
    with col2:
        st.subheader("How it Works")
        st.image("https://via.placeholder.com/400x250", caption="ML Recommendation System")
        st.write("""
        Our system uses:
        - Decision Tree classification for course category prediction
        - TF-IDF vectorization for processing skills data
        - Cosine similarity for job matching
        - PCA for feature reduction
        """)
    
    st.markdown("---")
    st.subheader("Quick Stats")
    
    try:
        student_data, job_data = load_data()
        
        if student_data is not None and job_data is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Courses", f"{len(student_data)}")
            
            with col2:
                st.metric("Course Categories", f"{student_data['Category'].nunique()}")
            
            with col3:
                st.metric("Available Job Titles", f"{len(job_data['Job Title'].unique())}")
    except:
        st.warning("Could not load statistics. Data files may be missing.")

# Predictions page
elif page == "Predictions":
    st.header("Make Predictions")
    
    # Load models and data
    models = load_models()
    student_data, _ = load_data()
    
    if models and student_data is not None:
        # Extract valid branch values from the dataset
        valid_branches = student_data['Branch'].unique().tolist()
        valid_course_types = student_data['Course Type'].unique().tolist()
        
        st.subheader("Student Profile")
        
        # Sample profiles button
        if st.button("Load Sample Profile"):
            # Randomly select a row from the dataset for a sample profile
            sample = student_data.sample(1).iloc[0]
            sample_branch = sample['Branch']
            sample_10th = sample['Percentage_10th']
            sample_12th = sample['Percentage_12th'] 
            sample_course_type = sample['Course Type']
            sample_skills = sample['Skills'][:100]  # Truncate skills for display
            
            st.session_state['branch'] = sample_branch
            st.session_state['percentage_10th'] = sample_10th
            st.session_state['percentage_12th'] = sample_12th
            st.session_state['course_type'] = sample_course_type
            st.session_state['skills'] = sample_skills
        
        col1, col2 = st.columns(2)
        
        with col1:
            branch = st.selectbox("Branch", 
                                 options=valid_branches,
                                 index=0 if 'branch' not in st.session_state else valid_branches.index(st.session_state['branch']))
            
            percentage_10th = st.slider("10th Percentage", 
                                       min_value=60, 
                                       max_value=100, 
                                       value=st.session_state.get('percentage_10th', 80))
            
            percentage_12th = st.slider("12th Percentage", 
                                       min_value=60, 
                                       max_value=100, 
                                       value=st.session_state.get('percentage_12th', 75))
        
        with col2:
            course_type = st.selectbox("Course Type", 
                                      options=valid_course_types,
                                      index=0 if 'course_type' not in st.session_state else valid_course_types.index(st.session_state['course_type']))
            
            skills = st.text_area("Skills", 
                                 value=st.session_state.get('skills', "python, data analysis, machine learning"))
        
        if st.button("Predict Course Category"):
            try:
                # Process input data
                skills_processed = skills.lower().replace(',', '')
                
                # Calculate derived features
                academic_score = 0.6 * percentage_12th + 0.4 * percentage_10th
                performance_gap = percentage_12th - percentage_10th
                skills_count = len(skills_processed.split())
                
                # Encode categorical features - with error handling for unseen labels
                try:
                    branch_encoded = models['label_encoders']['Branch'].transform([branch])[0]
                except ValueError:
                    st.error(f"Branch '{branch}' was not present in training data. Please select another branch.")
                    st.stop()
                    
                try:
                    course_type_encoded = models['label_encoders']['Course Type'].transform([course_type])[0]
                except ValueError:
                    st.error(f"Course Type '{course_type}' was not present in training data. Please select another course type.")
                    st.stop()
                
                # Create feature vector with basic features
                features = pd.DataFrame({
                    'Percentage_10th': [percentage_10th],
                    'Percentage_12th': [percentage_12th],
                    'Skills_Count': [skills_count],
                    'Academic_Score': [academic_score],
                    'Performance_Gap': [performance_gap],
                    'Course_Type_Encoded': [course_type_encoded],
                    'Branch_Encoded': [branch_encoded]
                })
                
                # Transform skills with TF-IDF
                skills_tfidf = models['tfidf'].transform([skills_processed]).toarray()
                tfidf_feature_names = models['tfidf'].get_feature_names_out()
                tfidf_df = pd.DataFrame(skills_tfidf, columns=tfidf_feature_names)
                
                # Combine features
                input_df = pd.concat([features.reset_index(drop=True), tfidf_df], axis=1)
                
                # Ensure all columns from training data are present
                # This step is crucial if the TF-IDF vectorizer doesn't recognize some words
                expected_columns = models['pca'].feature_names_in_
                for col in expected_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data
                input_df = input_df[expected_columns]
                
                # Apply PCA
                input_pca = models['pca'].transform(input_df)
                
                # Make prediction
                prediction = models['model'].predict(input_pca)[0]
                prediction_proba = models['model'].predict_proba(input_pca)[0]
                
                # Convert prediction to category name
                category_name = models['label_encoders']['Category'].inverse_transform([prediction])[0]
                
                # Display result
                st.success(f"Predicted Course Category: **{category_name}**")
                
                # Show prediction confidence
                max_proba = np.max(prediction_proba) * 100
                st.write(f"Confidence: {max_proba:.2f}%")
                
                # Predict matching jobs
                st.subheader("Matching Job Opportunities")
                
                # Find similar jobs using cosine similarity
                category_vector = models['job_vectorizer'].transform([category_name])
                similarities = cosine_similarity(category_vector, models['job_tfidf_matrix']).flatten()
                top_indices = np.argsort(similarities)[-4:][::-1]
                top_jobs = [models['job_titles'][i] for i in top_indices]
                
                # Display job matches
                for i, job in enumerate(top_jobs, 1):
                    st.write(f"**{i}.** {job}")
                
                # Visualization of job match confidence
                job_scores = [similarities[i] * 100 for i in top_indices]
                
                job_df = pd.DataFrame({
                    'Job': top_jobs,
                    'Match Score (%)': job_scores
                })
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(x='Match Score (%)', y='Job', data=job_df, palette='viridis', ax=ax)
                ax.set_xlabel('Match Confidence (%)')
                ax.set_ylabel('Job Title')
                ax.set_title('Job Match Confidence')
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
                st.write("Debug information:")
                st.write(f"Branch: {branch}")
                st.write(f"Course Type: {course_type}")
    else:
        st.warning("Models or data could not be loaded. Please check the files.")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    student_data, job_data = load_data()
    
    if student_data is not None:
        tab1, tab2 = st.tabs(["Course Data", "Job Data"])
        
        with tab1:
            st.subheader("Course Category Distribution")
            
            # Category distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            category_counts = student_data['Category'].value_counts().nlargest(10)
            sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis', ax=ax)
            ax.set_title('Top 10 Course Categories')
            ax.set_xlabel('Count')
            ax.set_ylabel('Category')
            st.pyplot(fig)
            
            # Course type distribution
            st.subheader("Course Type Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            course_type_counts = student_data['Course Type'].value_counts()
            plt.pie(course_type_counts, labels=course_type_counts.index, autopct='%1.1f%%', startangle=90,
                   colors=sns.color_palette('viridis', n_colors=len(course_type_counts)))
            plt.axis('equal')
            plt.title('Course Type Distribution')
            st.pyplot(fig)
            
            # Branch distribution
            st.subheader("Branch Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            branch_counts = student_data['Branch'].value_counts().nlargest(10)
            sns.barplot(x=branch_counts.values, y=branch_counts.index, palette='viridis', ax=ax)
            ax.set_title('Top 10 Branches')
            ax.set_xlabel('Count')
            ax.set_ylabel('Branch')
            st.pyplot(fig)
            
            # Academic performance analysis
            st.subheader("Academic Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(student_data['Percentage_10th'], kde=True, ax=ax)
                ax.set_title('10th Percentage Distribution')
                ax.set_xlabel('Percentage')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(student_data['Percentage_12th'], kde=True, ax=ax)
                ax.set_title('12th Percentage Distribution')
                ax.set_xlabel('Percentage')
                st.pyplot(fig)
        
        with tab2:
            if job_data is not None:
                st.subheader("Job Title Distribution")
                
                # Job title distribution
                fig, ax = plt.subplots(figsize=(10, 8))
                job_counts = job_data['Job Title'].value_counts().nlargest(15)
                sns.barplot(x=job_counts.values, y=job_counts.index, palette='viridis', ax=ax)
                ax.set_title('Top 15 Job Titles')
                ax.set_xlabel('Count')
                ax.set_ylabel('Job Title')
                st.pyplot(fig)
                
                # Display job data sample
                st.subheader("Job Data Sample")
                st.dataframe(job_data.head())
            else:
                st.warning("Job data could not be loaded.")
    else:
        st.warning("Data could not be loaded for analysis.")

# About page
elif page == "About":
    st.header("About This Project")
    
    st.write("""
    ## Project Overview
    
    This Course Category and Job Recommendation System is designed to help students make informed decisions about their educational and career paths.
    
    ### Features:
    - **Course Category Prediction**: Uses machine learning to predict the most suitable course category based on student profile and skills
    - **Job Opportunity Matching**: Suggests potential job opportunities that align with the predicted course category
    - **Data Visualization**: Provides insights into course distribution and job market trends
    
    ### Technologies Used:
    - **Python**: Core programming language
    - **Streamlit**: Web application framework
    - **Scikit-learn**: Machine learning algorithms and data processing
    - **Pandas & NumPy**: Data manipulation and analysis
    - **Matplotlib & Seaborn**: Data visualization
    
    ### Machine Learning Models:
    - **Decision Tree Classifier**: For course category prediction
    - **TF-IDF Vectorization**: For processing text data (skills)
    - **Principal Component Analysis (PCA)**: For dimensionality reduction
    - **Cosine Similarity**: For job matching
    """)
    
    st.markdown("---")
    
    st.write("Developed as part of an educational project to demonstrate machine learning applications in career guidance.")

# Add footer
st.markdown("---")
st.markdown("<p style='text-align: center'>© 2025 Course & Job Recommendation System</p>", unsafe_allow_html=True)