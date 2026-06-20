# Loan Approval Prediction - Installation & Setup Guide

## Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/LoanApproval_Prediction.git
cd LoanApproval_Prediction
```

### 2. Create and Activate Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the Model (First Time)

```bash
python train_model.py
```

This will:
- Load the dataset from `data/LoanApprovalPrediction.csv`
- Preprocess the data
- Train the Random Forest model
- Save the model to `models/random_forest_model.pkl`

### 5. Run the Web Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Project Structure

```
LoanApproval_Prediction/
├── src/                           # Core modules
│   ├── config.py                  # Configuration
│   ├── data_processor.py          # Data processing
│   ├── logger.py                  # Logging setup
│   ├── model_manager.py           # Model inference
│   └── utils.py                   # Utility functions
├── data/                          # Datasets
│   └── LoanApprovalPrediction.csv
├── models/                        # Trained models
│   └── random_forest_model.pkl
├── notebooks/                     # Jupyter notebooks
│   └── Loan_Approval_Analysis.ipynb
├── app.py                         # Streamlit application
├── train_model.py                 # Training script
└── requirements.txt               # Dependencies
```

## Usage

### Interactive Web Application

```bash
streamlit run app.py
```

Then:
1. Enter loan applicant details
2. Click "🔍 Predict Loan Approval"
3. View instant prediction with confidence metrics

### Python Script Usage

```python
from src.model_manager import get_model_manager
from src.data_processor import DataProcessor

# Initialize
model_manager = get_model_manager()
processor = DataProcessor()

# Process input
input_data = processor.process_input(
    gender='Male',
    married='Yes',
    dependents='1',
    education='Graduate',
    self_employed='No',
    applicant_income=75000,
    coapplicant_income=35000,
    loan_amount=180000,
    loan_amount_term=360,
    credit_history=1,
    property_area='Urban'
)

# Make prediction
prediction = model_manager.predict(input_data)
proba = model_manager.predict_proba(input_data)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {proba[0]}")
```

## Troubleshooting

### Issue: Model file not found
**Solution**: Run `python train_model.py` to train the model first

### Issue: Port 8501 already in use
**Solution**: Use `streamlit run app.py --server.port 8502`

### Issue: Missing dependencies
**Solution**: 
```bash
pip install -r requirements.txt
pip install --force-reinstall -r requirements.txt
```

### Issue: Import errors
**Solution**: Ensure you're in the correct directory and virtual environment is activated

## Development

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ app.py train_model.py
```

### Linting

```bash
flake8 src/ app.py train_model.py
```

## Uninstall

To remove the project:

1. Deactivate virtual environment: `deactivate`
2. Remove virtual environment: `rm -rf venv`
3. Remove project directory: `rm -rf LoanApproval_Prediction`

## Support

For issues or questions:
- Check README.md for comprehensive documentation
- Review notebooks for examples
- Open an issue on GitHub

---

**Happy predicting!** 🚀
