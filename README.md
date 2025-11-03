# YourCab - Cab Cancellation Prediction Web Application

A complete machine learning web application that predicts whether a cab booking will be cancelled or completed using Random Forest Classifier. Built with Flask, scikit-learn, and modern web technologies.

## 🎯 Features

- **Data Input Module**: Upload booking data in CSV format
- **Data Preprocessing**: Automatic handling of missing values, feature engineering (booking gap, peak hours, weekend flags)
- **ML Model Training**: Random Forest Classifier with 70/30 train-test split
- **Prediction Engine**: 
  - Single booking predictions with probability scores
  - Batch predictions for multiple bookings
- **Evaluation Dashboard**: 
  - Model accuracy, precision, recall, F1-score
  - Confusion matrix visualization
  - Feature importance analysis
  - Target: ≥90% accuracy

## 📋 Requirements

- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

## 🚀 Installation

1. **Clone the repository**:
```bash
cd /path/to/ankit
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Application

1. **Start the Flask server**:
```bash
python app.py
```

2. **Access the application**:
Open your browser and navigate to `http://localhost:5000`

### Using the Web Interface

#### 1. Train the Model
- On the home page, upload your training dataset (CSV format)
- The model will be trained automatically with a 70/30 train-test split
- View training metrics immediately after completion

#### 2. View Dashboard
- Navigate to the Dashboard page
- View model performance metrics (accuracy, precision, recall, F1-score)
- Analyze confusion matrix and feature importance
- Identify key factors influencing cancellations

#### 3. Make Predictions

**Single Prediction:**
- Navigate to the Predict page
- Fill in the booking details form
- Click "Predict" to get instant results with probability scores

**Batch Prediction:**
- Switch to the "Batch Prediction" tab
- Upload a CSV file with multiple bookings
- Download results with predictions for all bookings

## 📊 Dataset Format

Your CSV file should contain the following columns:

| Column | Description |
|--------|-------------|
| `booking_id` | Unique booking identifier |
| `user_id` | User/customer ID |
| `vehicle_model_id` | Vehicle model identifier |
| `travel_type_id` | Type of travel (1=Local, 2=Outstation, 3=Late Night) |
| `from_area_id` | Pickup area ID |
| `to_area_id` | Drop-off area ID |
| `from_date` | Trip start datetime (YYYY-MM-DD HH:MM:SS) |
| `to_date` | Trip end datetime (YYYY-MM-DD HH:MM:SS) |
| `from_lat` | Pickup latitude |
| `from_long` | Pickup longitude |
| `to_lat` | Drop-off latitude |
| `to_long` | Drop-off longitude |
| `car_cancellation` | Target variable (0=Completed, 1=Cancelled) |

**Sample data** is provided in `YourCabs.csv` with 100 sample bookings.

## 🔧 Feature Engineering

The application automatically creates the following derived features:

1. **Booking Gap**: Time difference between trip start and end (in minutes)
2. **Peak Hour Indicator**: Flags bookings during peak hours (7-9 AM, 5-8 PM)
3. **Weekend Flag**: Identifies weekend bookings (Saturday, Sunday)
4. **Time of Day**: Categorizes bookings into time slots (night, morning, afternoon, evening)
5. **Hour**: Hour of the day when trip starts

## 🎓 Model Details

- **Algorithm**: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 10
- **Training Split**: 70% training, 30% testing
- **Stratification**: Applied to maintain class distribution
- **Performance Target**: ≥90% accuracy

## 📁 Project Structure

```
ankit/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── YourCabs.csv           # Sample dataset
├── README.md              # This file
├── mvp.md                 # Project requirements
├── templates/             # HTML templates
│   ├── index.html         # Home page with model training
│   ├── dashboard.html     # Evaluation dashboard
│   └── predict.html       # Prediction interface
└── uploads/               # Uploaded files and results (auto-created)
```

## 🎯 Expected Outcomes

- Predict cancellation likelihood with ~93% accuracy
- Identify major influencing factors:
  - Booking gap duration
  - Travel type (local vs outstation vs late night)
  - Area IDs (pickup and drop-off locations)
  - Peak hours and weekend patterns
- Provide actionable insights for:
  - Reducing cancellations
  - Optimizing driver allocation
  - Enhancing customer experience

## 🛠️ Tech Stack

- **Backend**: Flask (Python web framework)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript (vanilla)

## 📝 API Endpoints

- `GET /` - Home page with model training
- `POST /train` - Train the model with uploaded CSV
- `GET /dashboard` - View evaluation metrics and visualizations
- `GET /predict` - Prediction interface
- `POST /predict` - Make single or batch predictions
- `GET /download/predictions` - Download batch prediction results
- `GET /sample-data` - Download sample dataset

## 🔍 Key Insights

The application provides insights into:

1. **Model Performance**: Comprehensive metrics including accuracy, precision, recall, and F1-score
2. **Feature Importance**: Which factors most influence cancellation predictions
3. **Prediction Confidence**: Probability scores for each prediction
4. **Data Patterns**: Confusion matrix showing prediction accuracy distribution

## 🎨 Screenshots

The web application features:
- Modern, responsive design with gradient backgrounds
- Interactive forms for data input
- Real-time prediction results with visual probability bars
- Professional data visualizations (confusion matrix, feature importance)
- Clean, intuitive navigation

## 🤝 Contributing

This is a MVP (Minimum Viable Product) implementation. Future enhancements could include:
- User authentication and session management
- Database integration for storing predictions
- Advanced visualizations and analytics
- Real-time prediction API
- Model versioning and A/B testing

## 📄 License

This project is created for educational and demonstration purposes.

## 👥 Authors

Created as part of the YourCab cancellation prediction system development.
