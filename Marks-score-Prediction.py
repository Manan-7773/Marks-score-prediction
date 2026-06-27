import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

url = "https://raw.githubusercontent.com/Manan-7773/Marks-score-prediction/main/student_multi.csv"
df = pd.read_csv(url)

print(df)

# Handling missing values

print(df.isnull().sum())

# Remove missing values (if any)
df.dropna(inplace=True)

# Label Encoding
course_encoder = LabelEncoder()
df["course_name"] = course_encoder.fit_transform(df["course_name"])

# Features and Target
x = df.drop("exam_score", axis=1)
y = df["exam_score"]

# Feature Scaling
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(x_train, y_train)


# User Input for Prediction


course = input("Enter Course Name (History, Math, Physics): ").title()
class_number = int(input("Enter Class Number: "))
study_hours = float(input("Enter Study Hours: "))
last_exam_score = int(input("Enter Last Exam Score: "))

# Encode course name



course = course_encoder.transform([course])

# Create DataFrame
new_data = pd.DataFrame({
    "course_name": [course[0]],
    "class_number": [class_number],
    "study_hours": [study_hours],
    "last_exam_score": [last_exam_score]
})

# Scale input data
new_data = scaler.transform(new_data)

# Prediction
predicted_score = model.predict(new_data)

print(f"\nPredicted Exam Score: {predicted_score[0]:.2f}")

import seaborn as sns
import matplotlib.pyplot as plt

# Scatter Plot
plt.figure(figsize=(6,4))
sns.scatterplot(x="study_hours", y="exam_score", data=df)
plt.title("Study Hours vs Exam Score")
plt.show()

# Bar Plot
plt.figure(figsize=(6,4))
sns.barplot(x="course_name", y="exam_score", data=df)
plt.title("Course vs Exam Score")
plt.show()

# Box Plot
plt.figure(figsize=(6,4))
sns.boxplot(y="exam_score", data=df)
plt.title("Exam Score Box Plot")
plt.show()

# Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()