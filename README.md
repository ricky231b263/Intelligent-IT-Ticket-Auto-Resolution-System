# Intelligent-IT-Ticket-Auto-Resolution-System

An AI-powered IT support ticket auto-resolution system that classifies enterprise support tickets, handles screenshot-based tickets using OCR, predicts issue types, and suggests instant solutions through a Streamlit web app.

---

## Project Overview
In enterprise environments, IT support teams receive a large number of repetitive support tickets such as VPN failures, Outlook login issues, printer problems, software crashes, and access-related complaints.  
This project automates the first level of ticket handling by predicting the **ticket category**, identifying the **most likely issue type**, and suggesting an **instant resolution**.

The system also supports **screenshot-based tickets** by extracting text from uploaded images using OCR and then running prediction on the extracted content.

---

## Features
- Automatic **IT ticket category classification**
- **Issue type prediction** based on ticket text
- **Instant resolution suggestion**
- **OCR support** for screenshot-only tickets
- **Streamlit web app** for interactive testing
- Built using **synthetic large-scale enterprise IT ticket data**

---

## Problem Statement
Manual IT helpdesk ticket triage takes time and often delays resolution for repetitive issues.  
The goal of this project is to build an intelligent system that can:

1. Read the user’s ticket text or screenshot
2. Predict the ticket category
3. Identify the issue type
4. Suggest a likely solution automatically

This reduces manual effort and improves response time for common support requests.

---

## Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **EasyOCR**
- **Joblib**

---

## Project Workflow

### 1. Synthetic Dataset Generation
A large synthetic enterprise IT support ticket dataset was generated containing ticket fields such as:
- ticket subject
- ticket description
- category
- issue type
- priority
- resolution
- screenshot availability
- OCR/log text support

### 2. Text Preprocessing
Ticket subject and description were combined into a single input text representation for model training.

### 3. Category Classification
A machine learning model was trained to classify tickets into categories such as:
- Email
- Network
- Hardware
- Software
- Security
- Printer
- Database
- Access

### 4. Issue Type Prediction
Instead of training an extremely heavy multi-class issue model over thousands of issue labels, the project uses:
- **category prediction first**
- then **issue lookup / retrieval inside that category**
- and finally **resolution mapping**

This keeps the system faster and easier to deploy.

### 5. OCR-based Screenshot Handling
If the user uploads a screenshot:
- EasyOCR extracts text from the image
- extracted text is merged with the ticket text
- prediction is performed on the final combined input

### 6. Streamlit Deployment
A Streamlit web application is used to:
- enter ticket text manually
- upload a screenshot
- predict category
- predict issue type
- display a suggested resolution

---

## Folder Structure
```bash
Intelligent-IT-Ticket-Auto-Resolution-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── DATASET.ipynb
├── model-fit.ipynb
└── saved_models/
    ├── category_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── category_encoder.pkl
    ├── issue_resolution_map.pkl
    └── category_issue_lookup.pkl
