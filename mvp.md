MVP (Minimum Viable Product) — YourCab: Cab Cancellation Prediction
Objective

Develop a functional machine learning prototype that predicts whether a cab booking will be cancelled or completed, using historical booking data from the YourCab platform.

Core Features

Data Input Module

Accepts booking details in CSV or database format.

Includes fields like booking time, trip start time, travel type, vehicle model, and area IDs.

Data Preprocessing

Handles missing values using mean/mode imputation.

Converts categorical features (like travel type, booking source) into numerical form.

Generates derived features:

Booking Gap (time between booking and trip start)

Peak Hour Indicator

Weekend Flag

Model Training

Uses the Random Forest Classifier as the base model for prediction.

Trains on preprocessed booking data.

Splits dataset into 70% training and 30% testing sets.

Prediction Engine

Classifies new bookings as:

0 → Completed

1 → Cancelled

Returns prediction probability for better interpretability.

Evaluation Dashboard

Displays model accuracy, precision, recall, and F1-score.

Includes confusion matrix visualization.

Target benchmark: ≥90% accuracy.

Expected Outcome

Predict cancellation likelihood for new bookings with ~93% accuracy.

Identify major influencing factors like booking gap, travel type, and area IDs.

Provide actionable insights for:

Reducing cancellations.

Optimizing driver allocation.

Enhancing customer experience.

Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

Environment: Jupyter Notebook / Google Colab

Algorithm: Random Forest Classifier

Deliverables (for MVP version)

Jupyter Notebook with data preprocessing, model training, and evaluation.

CSV input file (YourCabs.csv) for testing predictions.

Output report showing key metrics and important features.

Simple interface (CLI or minimal web dashboard) to upload booking data and get predictions.
