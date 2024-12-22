# Credit Card Approval Prediction

This project utilizes machine learning techniques to predict credit card approval decisions based on applicant data and credit history.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The objective of this project is to automate the assessment of credit card applications by developing a predictive model. Such a model assists financial institutions in streamlining the approval process and making informed decisions.

## Dataset

The project employs two primary datasets:

- **application_record.csv**: Contains applicant information, including ID, gender, and occupation.
- **credit_record.csv**: Includes credit history records for each applicant.

These datasets are preprocessed and merged to create a comprehensive dataset for model training.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zyad-ashraf-amar/Credit-Card-Approval.git
   cd Credit-Card-Approval
2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt

## Usage

The project includes a Jupyter Notebook that details the data preprocessing, exploratory data analysis, feature engineering, and model development:

- **credit-card-approval-prediction.ipynb**: Open this notebook to follow the step-by-step process of building the predictive model.

To run the notebook:

     ```bash
     jupyter notebook credit-card-approval-prediction.ipynb

## Model Training
The notebook guides you through:

- Loading and exploring the datasets.
- Preprocessing data, including handling missing values and encoding categorical variables.
- Feature selection and engineering.
- Training various machine learning models and evaluating their performance.

## Deployment
An application is provided to deploy the trained model:

- app.py: This script sets up a web interface for users to input applicant data and receive approval predictions.
To run the application:

      ```bash
      streamlit app.py
Then, navigate to http://localhost:5000 in your browser to use the application.


## Contact
For any inquiries or feedback, please contact:

- Name: Zyad Ashraf Amar
- Email: zyadashrafamar@gmail.com
- GitHub: zyad-ashraf-amar
