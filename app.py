from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import os

app = Flask(__name__)

# Load and preprocess dataset
def load_and_preprocess_data():
    # Change to read Excel file instead of CSV
    df = pd.read_excel('disease_dataset.xlsx')  # Adjust to the path of your Excel file
    label_encoder = LabelEncoder()
    df['Disease'] = label_encoder.fit_transform(df['Disease'])
    
    # Create binary features for symptoms
    all_symptoms = set()
    for symptoms in df['Symptoms']:
        all_symptoms.update(symptoms.split(', '))
    all_symptoms = sorted(list(all_symptoms))
    
    for symptom in all_symptoms:
        df[symptom] = df['Symptoms'].apply(lambda x: 1 if symptom in x else 0)
    
    X = df.drop(columns=['Disease', 'Symptoms']).values
    y = to_categorical(df['Disease'])
    return X, y, all_symptoms, label_encoder

# Build and train model
def build_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data and train the model
X, y, all_symptoms, label_encoder = load_and_preprocess_data()
model = build_model(X.shape[1], y.shape[1])
model.fit(X, y, epochs=30, batch_size=8, verbose=0)

# Home route
@app.route('/')
def index():
    return render_template('index.html', symptoms=all_symptoms)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the symptoms from the user input
    user_symptoms = request.form['symptoms'].split(',')  # Split the input string by commas
    user_symptoms = [symptom.strip() for symptom in user_symptoms]  # Remove extra spaces
    
    # Create the input vector for the model (1 if the symptom is present, else 0)
    input_vector = [1 if symptom in user_symptoms else 0 for symptom in all_symptoms]
    input_vector = np.array([input_vector])
    
    # Make the prediction
    prediction = model.predict(input_vector)
    predicted_class = np.argmax(prediction)
    disease_name = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get the confidence (probability) of the predicted disease
    confidence = prediction[0][predicted_class] * 100
    
    # Pass the result to the results.html page
    return render_template('result.html', disease_name=disease_name, confidence=confidence)

@app.route('/book_appointment/<disease_name>')
def book_appointment(disease_name):
    return render_template('book_appointment.html', disease_name=disease_name)

@app.route('/submit_appointment', methods=['POST'])
def submit_appointment():
    # Retrieve form data (no 'address' field now)
    name = request.form['name']
    contact = request.form['contact']
    date = request.form['date']
    time = request.form['time']
    disease_name = request.form['disease_name']
    
    # Prepare the appointment data
    appointment_data = {
        'Name': name,
        'Contact': contact,
        'Date': date,
        'Time': time,
        'Disease': disease_name,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save the appointment data to Excel (as before)
    file_path = 'appointment_bookings.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        new_row = pd.DataFrame([appointment_data])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([appointment_data])
    
    df.to_excel(file_path, index=False, engine='openpyxl')

    # Optionally print the appointment details
    print(f"Appointment booked for {name} ({contact}) on {date} at {time} for {disease_name}")
    
    return render_template('appointment_success.html', disease_name=disease_name)




@app.route('/prescription/<disease_name>')
def prescription(disease_name):
    # Define some sample tablet prescriptions based on the disease
    prescriptions = {
        'Flu': ['Paracetamol', 'Ibuprofen', 'Cough Syrup'],
        'Cold': ['Vitamin C', 'Cough Syrup', 'Decongestant'],
        'COVID-19': ['Paracetamol', 'Ivermectin', 'Doxycycline'],
        'Malaria': ['Chloroquine', 'Artemisinin'],
        'Dengue': ['Paracetamol'],
        'Chickenpox': ['Acyclovir', 'Paracetamol'],
        'Asthma': ['Salbutamol', 'Budesonide'],
        'Pneumonia': ['Amoxicillin', 'Azithromycin']
    }
    
    tablets = prescriptions.get(disease_name, [])
    return render_template('prescription.html', disease_name=disease_name, tablets=tablets)

@app.route('/book_test/<disease_name>')
def book_test(disease_name):
    return render_template('tests.html', disease_name=disease_name)

# Route to handle Test Booking Form submission
@app.route('/submit_test', methods=['POST'])
def submit_test():
    # Retrieve form data
    name = request.form['name']
    contact = request.form['contact']
    address = request.form['address']
    disease_name = request.form['disease_name']
    
    # Prepare the test booking data
    test_booking_data = {
        'Name': name,
        'Contact': contact,
        'Address': address,
        'Disease': disease_name,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Specify the file path for storing test booking data
    file_path = 'test_bookings.xlsx'
    
    # Check if the file exists, if not create it and save the data
    if os.path.exists(file_path):
        # If the file exists, read the existing data and add the new row
        df = pd.read_excel(file_path)
        new_row = pd.DataFrame([test_booking_data])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        # If the file does not exist, create a new DataFrame and save it
        df = pd.DataFrame([test_booking_data])
    
    # Save the updated DataFrame to the Excel file
    df.to_excel(file_path, index=False, engine='openpyxl')

    # Print the test booking details (optional for debugging)
    print(f"Test booked for {name} ({contact}) at {address} for {disease_name}")
    
    # Render the success page after saving the data
    return render_template('test_success.html', disease_name=disease_name)



if __name__ == '__main__':
    app.run(debug=True)
