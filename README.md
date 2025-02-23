# Service Request Segmentation

## Project Overview
This project clusters **service requests (SRs)** into groups using **unsupervised learning (K-Means clustering)**. 
The goal is to **automatically categorize service requests** and assign them to the right response team, reducing workload.

## Features
- **TF-IDF for text representation** of service requests.
- **K-Means Clustering** to segment similar SRs.
- **Automated workflow** for routing SRs to appropriate teams.

## Project Structure
```
Service_Request_Segmentation/
│── data/                 # Raw and processed service request data
│── notebooks/            # Jupyter notebooks for EDA and model experiments
│── src/                  # Source code for feature extraction and model training
│   ├── features/         # TF-IDF feature extraction script
│   ├── models/           # Unsupervised clustering model training
│── reports/              # Model performance reports
│── README.md             # Documentation
```

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Extract text features from service request descriptions:
   ```bash
   python src/features/extract_features.py
   ```
3. Train the clustering model:
   ```bash
   python src/models/train_model.py
   ```

## License
Open-source project for research and educational purposes.

