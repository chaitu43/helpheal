import pandas as pd

# Create a dictionary with sample data
data = {
    'Disease': [
        'Flu', 'Cold', 'COVID-19', 'Malaria', 'Dengue', 
        'Chickenpox', 'Asthma', 'Pneumonia'
    ],
    'Symptoms': [
        'Fever, Cough, Sore Throat, Headache',
        'Runny Nose, Sneezing, Cough, Headache',
        'Fever, Cough, Shortness of Breath, Fatigue',
        'Fever, Chills, Sweating, Headache, Nausea',
        'Fever, Rash, Headache, Pain Behind Eyes',
        'Rash, Fever, Tiredness, Itching',
        'Shortness of Breath, Wheezing, Cough, Chest Tightness',
        'Cough, Fever, Shortness of Breath, Chest Pain'
    ]
}

# Convert dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame as an Excel file
df.to_excel('disease_dataset.xlsx', index=False)

print("Dataset created and saved as 'disease_dataset.xlsx'")
