# Personality Prediction: Extrovert vs. Introvert

## Overview

This project explores predicting personality traits—specifically whether someone is an extrovert or introvert—based on behavioral and social features. Using a dataset containing information such as time spent alone, social event attendance, stage fright, and social media activity, the project builds and evaluates machine learning models to classify personality types.

## Dataset

- **Size:** 2,900 entries with 8 columns
- **Features:**
  - `Time_spent_Alone`: Hours spent alone daily (0–11)
  - `Stage_fear`: Presence of stage fright (Yes/No)
  - `Social_event_attendance`: Frequency of social events (0–10)
  - `Going_outside`: Frequency of going outside (0–7)
  - `Drained_after_socializing`: Feeling drained after socializing (Yes/No)
  - `Friends_circle_size`: Number of close friends (0–15)
  - `Post_frequency`: Social media post frequency (0–10)
  - `Personality`: Target variable (Extrovert/Introvert)

## Project Steps

1. **Data Preprocessing:**
   - Handle missing values by imputing with column means for numerical features.
   - Encode categorical variables (`Stage_fear`, `Drained_after_socializing`, `Personality`) into numeric format.
   
2. **Train-Test Split:**
   - Split the dataset into training (80%) and testing (20%) sets to evaluate model generalization.

3. **Model Training and Evaluation:**
   - Trained two classifiers:
     - Logistic Regression
     - Random Forest
   - Evaluated model performance using accuracy, precision, recall, F1-score, and confusion matrix.

## Results

- Logistic Regression achieved **~91.9% accuracy** on the test set.
- Random Forest achieved **~90.9% accuracy** on the test set.
- Both models performed comparably well, with Logistic Regression slightly ahead in accuracy.

## Technologies Used

- Python
- pandas (for data manipulation)
- scikit-learn (for machine learning models and evaluation)

## How to Run

1. Clone the repository.
2. Install dependencies via pip:

   ```bash
   pip install pandas scikit-learn


# Author: Spenser Williams
