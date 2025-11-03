"""
YourCab Cancellation Prediction Web Application
This Flask web app predicts cab booking cancellations using Random Forest Classifier
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and metrics
model = None
metrics = {}
feature_importance = {}
confusion_mat = None

def preprocess_data(df):
    """
    Preprocess the booking data:
    - Handle missing values
    - Create derived features (booking_gap, peak_hour, weekend_flag)
    - Convert categorical to numerical
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Convert date columns to datetime
    data['from_date'] = pd.to_datetime(data['from_date'])
    data['to_date'] = pd.to_datetime(data['to_date'])
    
    # Create derived features
    # 1. Booking Gap (assuming booking happens just before from_date in this dataset)
    data['booking_gap_minutes'] = (data['to_date'] - data['from_date']).dt.total_seconds() / 60
    
    # 2. Peak Hour Indicator (7-9 AM and 5-8 PM)
    data['hour'] = data['from_date'].dt.hour
    data['peak_hour'] = ((data['hour'] >= 7) & (data['hour'] <= 9) | 
                         (data['hour'] >= 17) & (data['hour'] <= 20)).astype(int)
    
    # 3. Weekend Flag (Saturday=5, Sunday=6)
    data['weekend_flag'] = (data['from_date'].dt.dayofweek >= 5).astype(int)
    
    # 4. Time of day category
    data['time_of_day'] = pd.cut(data['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=[0, 1, 2, 3], 
                                  include_lowest=True).astype(int)
    
    # Handle missing values using mean/mode imputation
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data[col].isnull().any():
            data[col].fillna(data[col].mean(), inplace=True)
    
    # Select features for training
    feature_columns = ['vehicle_model_id', 'travel_type_id', 'from_area_id', 'to_area_id',
                      'booking_gap_minutes', 'peak_hour', 'weekend_flag', 'time_of_day', 'hour']
    
    return data, feature_columns

def train_model(csv_file_path):
    """
    Train the Random Forest Classifier model on the provided dataset
    """
    global model, metrics, feature_importance, confusion_mat
    
    # Load data
    df = pd.read_csv(csv_file_path)
    
    # Preprocess data
    data, feature_columns = preprocess_data(df)
    
    # Prepare features and target
    X = data[feature_columns]
    y = data['car_cancellation']
    
    # Split data (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall': round(recall_score(y_test, y_pred) * 100, 2),
        'f1_score': round(f1_score(y_test, y_pred) * 100, 2),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_samples': len(df)
    }
    
    # Feature importance
    importance_dict = dict(zip(feature_columns, model.feature_importances_))
    feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    # Confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return metrics, feature_importance, confusion_mat

def generate_confusion_matrix_plot():
    """
    Generate confusion matrix visualization as base64 image
    """
    if confusion_mat is None:
        return None
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Completed', 'Cancelled'],
                yticklabels=['Completed', 'Cancelled'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Cab Cancellation Prediction')
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def generate_feature_importance_plot():
    """
    Generate feature importance visualization as base64 image
    """
    if not feature_importance:
        return None
    
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importances = list(feature_importance.values())
    
    plt.barh(features, importances, color='steelblue')
    plt.xlabel('Importance')
    plt.title('Feature Importance - Random Forest')
    plt.tight_layout()
    
    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

@app.route('/')
def index():
    """
    Home page
    """
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    """
    Train the model with uploaded CSV file
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400
    
    try:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'training_data.csv')
        file.save(filepath)
        
        # Train model
        train_model(filepath)
        
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """
    Display evaluation dashboard with metrics and visualizations
    """
    if model is None:
        return render_template('dashboard.html', trained=False)
    
    # Generate plots
    cm_plot = generate_confusion_matrix_plot()
    fi_plot = generate_feature_importance_plot()
    
    return render_template('dashboard.html',
                         trained=True,
                         metrics=metrics,
                         feature_importance=feature_importance,
                         cm_plot=cm_plot,
                         fi_plot=fi_plot)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Make predictions on new data
    """
    if request.method == 'GET':
        return render_template('predict.html', model_trained=(model is not None))
    
    if model is None:
        return jsonify({'error': 'Model not trained yet'}), 400
    
    try:
        if 'file' in request.files and request.files['file'].filename:
            # Batch prediction from CSV
            file = request.files['file']
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Only CSV files are allowed'}), 400
            
            # Read CSV
            df = pd.read_csv(file)
            
            # Preprocess
            data, feature_columns = preprocess_data(df)
            X = data[feature_columns]
            
            # Predict
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            # Add results to dataframe
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['prediction_label'] = results_df['prediction'].map({0: 'Completed', 1: 'Cancelled'})
            results_df['cancellation_probability'] = probabilities[:, 1]
            
            # Save results
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
            results_df.to_csv(output_path, index=False)
            
            # Calculate summary
            total = len(predictions)
            cancelled = sum(predictions)
            completed = total - cancelled
            
            return jsonify({
                'success': True,
                'type': 'batch',
                'total': total,
                'cancelled': int(cancelled),
                'completed': int(completed),
                'download_url': '/download/predictions'
            })
        
        else:
            # Single prediction from form
            data = request.form
            
            # Create dataframe from form data
            input_data = {
                'booking_id': [1],
                'user_id': [int(data.get('user_id', 0))],
                'vehicle_model_id': [int(data.get('vehicle_model_id'))],
                'travel_type_id': [int(data.get('travel_type_id'))],
                'from_area_id': [int(data.get('from_area_id'))],
                'to_area_id': [int(data.get('to_area_id'))],
                'from_date': [data.get('from_date')],
                'to_date': [data.get('to_date')],
                'from_lat': [float(data.get('from_lat', 0))],
                'from_long': [float(data.get('from_long', 0))],
                'to_lat': [float(data.get('to_lat', 0))],
                'to_long': [float(data.get('to_long', 0))],
                'car_cancellation': [0]  # Placeholder
            }
            
            df = pd.DataFrame(input_data)
            
            # Preprocess
            processed_data, feature_columns = preprocess_data(df)
            X = processed_data[feature_columns]
            
            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            
            return jsonify({
                'success': True,
                'type': 'single',
                'prediction': int(prediction),
                'prediction_label': 'Cancelled' if prediction == 1 else 'Completed',
                'probability': {
                    'completed': round(float(probability[0]) * 100, 2),
                    'cancelled': round(float(probability[1]) * 100, 2)
                }
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/predictions')
def download_predictions():
    """
    Download prediction results
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name='predictions.csv')
    else:
        return jsonify({'error': 'No predictions file found'}), 404

@app.route('/sample-data')
def sample_data():
    """
    Download sample CSV file
    """
    sample_path = 'YourCabs.csv'
    if os.path.exists(sample_path):
        return send_file(sample_path, as_attachment=True, download_name='YourCabs_sample.csv')
    else:
        return jsonify({'error': 'Sample data not found'}), 404

if __name__ == '__main__':
    # Train model with default data if available
    if os.path.exists('YourCabs.csv'):
        print("Training model with default dataset...")
        train_model('YourCabs.csv')
        print(f"Model trained! Accuracy: {metrics['accuracy']}%")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
