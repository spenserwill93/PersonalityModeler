import pandas as pd
#Load and inspect dataset
df = pd.read_csv("personality_dataset.csv")

print(df.head())
print(df.info())

#Define column groups
num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
    'Friends_circle_size', 'Post_frequency']
cat_cols = ['Stage_fear', 'Drained_after_socializing']

#Fill in numeric columns with median
for col in num_cols:
    median = df[col].median()
    df[col].fillna(median, inplace =True)

#Fill in categorical columns with mode
for col in cat_cols:
    mode = df[col].mode()[0]
    df[col].fillna(mode, inplace= True)

#Check that missing values are handled
print("Missing Values: ")
print(df.isnull().sum())

from sklearn.preprocessing import LabelEncoder

#Initialze the label encoder
le = LabelEncoder()

#Encode Yes/No columns
df['Stage_fear'] = le.fit_transform(df["Stage_fear"]) #Yes =1, No = 0
df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing']) #Yes =1, No = 0

#Encode the 'target' column
df['Personality'] = df['Personality'].map({'Introvert' : 0, 'Extrovert': 1})

#Confirm encoding
print("\nUnique values after encoding:")
print("Stage_fear:", df['Stage_fear'].unique())
print("Drained_after_socializing: ", df['Drained_after_socializing'].unique())
print("Personality: ", df['Personality'].unique())

from sklearn.model_selection import train_test_split

#Set features
X = df.drop('Personality', axis=1)

#Set target variable
y = df['Personality']

#Split the dataset (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

#Check on shape
print("Training set:", X_train.shape)
print("Test set:", X_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Initialize the model
model = LogisticRegression(max_iter = 1000, random_state = 42)

#Train the model
model.fit(X_train, y_train)

#Predict on test set
y_pred = model.predict(X_test)

#Evaluate the results
print("Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the results
print("Random Forest Accuracy on test set:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))