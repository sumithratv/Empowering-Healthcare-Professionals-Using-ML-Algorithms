import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def FilteringFeaturesAndTarget():
    df = pd.read_csv('diabetic_data.csv')
    columns_to_filter = ['encounter_id', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'medical_specialty', 'num_lab_procedures', 'num_medications', 'number_emergency', 'metformin','repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed','readmitted']
    filtered_columns = df.loc[:, columns_to_filter]

    # Handling duplicates
    filtered_columns.drop_duplicates() # Remove duplicate rows

    # Cleaning inconsistent data
    filtered_columns['medical_specialty'] = filtered_columns['medical_specialty'].str.lower()  # Convert text to lowercase
    filtered_columns['metformin'] = filtered_columns['metformin'].str.lower()
    filtered_columns['repaglinide'] = filtered_columns['repaglinide'].str.lower()
    filtered_columns['nateglinide'] = filtered_columns['nateglinide'].str.lower()
    filtered_columns['chlorpropamide'] = filtered_columns['chlorpropamide'].str.lower()
    filtered_columns['glimepiride'] = filtered_columns['glimepiride'].str.lower()
    filtered_columns['acetohexamide'] = filtered_columns['acetohexamide'].str.lower()
    filtered_columns['glipizide'] = filtered_columns['glipizide'].str.lower()
    filtered_columns['glyburide'] = filtered_columns['glyburide'].str.lower()
    filtered_columns['tolbutamide'] = filtered_columns['tolbutamide'].str.lower()
    filtered_columns['pioglitazone'] = filtered_columns['pioglitazone'].str.lower()
    filtered_columns['rosiglitazone'] = filtered_columns['rosiglitazone'].str.lower()
    filtered_columns['acarbose'] = filtered_columns['acarbose'].str.lower()
    filtered_columns['miglitol'] = filtered_columns['miglitol'].str.lower()
    filtered_columns['troglitazone'] = filtered_columns['troglitazone'].str.lower()
    filtered_columns['tolazamide'] = filtered_columns['tolazamide'].str.lower()
    filtered_columns['examide'] = filtered_columns['examide'].str.lower()
    filtered_columns['citoglipton'] = filtered_columns['citoglipton'].str.lower()
    filtered_columns['insulin'] = filtered_columns['insulin'].str.lower()
    filtered_columns['glyburide-metformin'] = filtered_columns['glyburide-metformin'].str.lower()
    filtered_columns['glipizide-metformin'] = filtered_columns['glipizide-metformin'].str.lower()
    filtered_columns['glimepiride-pioglitazone'] = filtered_columns['glimepiride-pioglitazone'].str.lower()
    filtered_columns['metformin-rosiglitazone'] = filtered_columns['metformin-rosiglitazone'].str.lower()
    filtered_columns['metformin-pioglitazone'] = filtered_columns['metformin-pioglitazone'].str.lower()
    filtered_columns['change'] = filtered_columns['change'].str.lower()
    filtered_columns['diabetesMed'] = filtered_columns['diabetesMed'].str.lower()
    filtered_columns['readmitted'] = filtered_columns['readmitted'].str.lower()

    # Removing the special value
    filtered_columns = filtered_columns.dropna()
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != '?']
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != 'neurophysiology']
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != 'sportsmedicine']
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != 'psychiatry-addictive']
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != 'speech']
    filtered_columns = filtered_columns[filtered_columns['medical_specialty'] != 'perinatology']

    filtered_columns = filtered_columns[filtered_columns['chlorpropamide'] != 'down']
    filtered_columns = filtered_columns[filtered_columns['acarbose'] != 'down']
    filtered_columns = filtered_columns[filtered_columns['glipizide-metformin'] != 'steady']
    filtered_columns = filtered_columns[filtered_columns['metformin-pioglitazone'] != 'steady']

    # filtered_columns = filtered_columns.dropna()
    # filtered_columns = filtered_columns.loc[filtered_columns['diag_1'].str.startswith('V')]

    # Save the cleaned data to a new file
    filtered_columns.to_csv('cleaned_data.csv', index=False)

def DataPreparation(cleanedFileName):
    df = pd.read_csv(cleanedFileName)

    # Define the categorical and numerical features
    categorical_features = ['age', 'medical_specialty', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
                            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'examide',
                            'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
                            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change',
                            'diabetesMed']
    numerical_features = ['encounter_id', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                          'time_in_hospital', 'num_lab_procedures', 'num_medications']

    # Preprocessing steps for categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', 'passthrough', numerical_features)
        ])

    # Split your data into X (features) and y (target)
    X = df.drop('number_emergency', axis=1)
    y = df['number_emergency']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    return preprocessor,X_train, X_test, y_train, y_test