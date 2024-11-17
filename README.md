# NeuroCare
**Parkinson's Disease Prediction Dashboard**

This project is a Streamlit-based web application that predicts the likelihood of Parkinson's disease using machine learning models. The dashboard is interactive and allows users to explore feature correlations, visualize data distributions, select features, and choose a machine learning model for prediction.

**Features**

🌟** Interactive Dashboard**

Feature Selection: Choose specific features to include in the prediction model.
Model Selection: Select between Support Vector Machine (SVM) and Random Forest for prediction.
Dynamic Visualizations:
Correlation Heatmap: Understand feature relationships with the target variable (status).
Data Distribution Plots: View distributions of selected features, categorized by the target variable.
🧪 Prediction Capabilities
Input custom values for selected features.
Predict whether the person has Parkinson's disease based on the input.
Display prediction results dynamically.
⚙️ Model Performance
Training and test accuracy are shown to help understand model performance.
Dataset
The application uses the Parkinson’s Disease Dataset, which contains biomedical voice measurements from people with Parkinson's disease and healthy individuals. Key details:

status is the target variable (1 for Parkinson's disease, 0 otherwise).
Other features are voice-related measurements such as jitter, shimmer, and others.
Installation and Setup
Prerequisites
Ensure you have Python 3.8+ installed along with the following libraries:

streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
Steps
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo-name/parkinsons-disease-dashboard.git
cd parkinsons-disease-dashboard
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the app:

bash
Copy code
streamlit run parkinsons_dashboard.py
Open the link provided by Streamlit in your browser.

File Structure
bash
Copy code
parkinsons-disease-dashboard/
├── parkinsons_dashboard.py    # Main Streamlit app script
├── parkinsons_datset.csv      # Parkinson’s dataset
├── README.md                  # Project documentation
├── requirements.txt           # Required Python libraries
Usage
Feature Selection:

Choose the features to include in the prediction model.
View how selected features correlate with the target variable via the heatmap.
Model Selection:

Select a machine learning model: SVM or Random Forest.
Make a Prediction.

Input values for the selected features.
Click the "Predict" button to get the prediction result.
Explore the Data:

View dynamic plots to understand feature distributions.
![image](https://github.com/user-attachments/assets/cd987cc8-c78b-45b1-8af0-7953987da96c)
