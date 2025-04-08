# **Student Job & Course Recommendation System**  

## **Overview**  
The **Student Job & Course Recommendation System** helps students explore **career opportunities** and **skill-enhancing courses** tailored to their profiles. By analyzing a student’s **academic background, skills, and career aspirations**, the system suggests relevant **job postings** and **online courses** to improve their employability.  

## **Features**  
- **Personalized Recommendations** – Matches students to jobs & courses based on their skills and other features.  
- **Machine Learning-Powered Matching** – Uses **SVM, Decision Tree, and Random Forest** models for precise recommendations.  
- **Data-Driven Insights** – Identifies trends in **hiring demands, skill gaps, and career trajectories**.  
- **Web-Based Interface** – Provides an website dashboard for student interaction.  

## **Technology Stack**  
- **Backend** – Python, Flask  
- **Frontend** – HTML, CSS, JavaScript  
- **Data Analysis** – Pandas, Scikit-learn, Matplotlib, Seaborn  
- **ML Models** – Support Vector Classifier (SVC), Decision Tree, Random Forest  

## **Datasets Used**  
The system integrates data from three sources:  
1. **Student Data (`Studentdata.csv`)** – Academic performance, skills, and career goals.  
2. **Course Data (`Online_Courses.csv`)** – Course titles, categories, and required skills.  
3. **Job Data (`Job_Postings.csv`)** – Job titles, required skills, locations, and companies.  

## **Insights from Exploratory Data Analysis (EDA)**  
- **Top Career Aspirations:** ML Engineer, Web Developer, Data Scientist.  
- **Most Sought-After Skills:** Python, SQL, Machine Learning, Data Analysis.  
- **High-Demand Job Locations:** California, New York, Chicago, Texas.  
- **Industry Trends:** Companies like **Toptal, Jobot, and Perficient** are frequent recruiters.  

## **Key Methodologies**  
- **TF-IDF Vectorization** – Converts skills into numerical format for better matching.  
- **Feature Engineering** – Constructs an **Academic Performance Score** to refine recommendations.  
- **Categorical Encoding** – Transforms categorical data (e.g., course type, branch) for ML models.  

## **Project Roadmap**  
| Phase | Timeline |  
| --- | --- |  
| Model Optimization | Apr 6-11 |  
| Web Development | Apr 12-17 |  
| Model Integration | Apr 18-21 |  
| Final Testing & Submission | Apr 22-May 23 |  

## **Contributor**  
- **Sashank Boppana**  
- **Tejesh Boppana**  