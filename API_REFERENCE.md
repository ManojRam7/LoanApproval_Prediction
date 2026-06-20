# API Reference

## Module: src.config

Configuration management for the Loan Approval Prediction system.

### Constants

```python
MODEL_CONFIG          # Random Forest model parameters
FEATURE_NAMES         # List of feature names in order
ENCODING_MAPPINGS     # Categorical variable encoding mappings
TRAIN_TEST_SPLIT_RATIO  # Train/test split ratio (0.4)
```

### Paths

```python
PROJECT_ROOT         # Root project directory
DATA_DIR            # Data directory path
RAW_DATA_PATH       # Training dataset path
MODELS_DIR          # Models directory path
MODEL_PATH          # Trained model path
LOGS_DIR            # Logs directory path
```

---

## Module: src.model_manager

Model loading and inference management.

### Class: ModelManager

```python
ModelManager(model_path: str | Path = MODEL_PATH)
```

**Methods:**

- `predict(X: pd.DataFrame) -> np.ndarray`
  - Make predictions using the loaded model
  - Returns: prediction results (0 or 1)

- `predict_proba(X: pd.DataFrame) -> np.ndarray`
  - Get prediction probabilities
  - Returns: probability array

- `get_model_info() -> dict`
  - Get model information and parameters
  - Returns: dict with model details

### Function: get_model_manager()

```python
get_model_manager(model_path: str | Path = MODEL_PATH) -> ModelManager
```

Get or create singleton ModelManager instance.

**Example:**

```python
from src.model_manager import get_model_manager

manager = get_model_manager()
predictions = manager.predict(input_data)
```

---

## Module: src.data_processor

Data processing and transformation.

### Class: DataProcessor

```python
DataProcessor()
```

**Methods:**

- `process_input(**kwargs) -> pd.DataFrame`
  - Process and encode user input for predictions
  - Parameters: gender, married, dependents, education, self_employed, applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history, property_area
  - Returns: processed DataFrame

- `validate_inputs(**kwargs) -> bool`
  - Validate user input data
  - Raises: ValueError if validation fails
  - Returns: True if valid

- `load_and_prepare_data(file_path: str) -> pd.DataFrame`
  - Load and prepare raw CSV data
  - Parameters: file_path - path to CSV file
  - Returns: prepared DataFrame

**Example:**

```python
from src.data_processor import DataProcessor

processor = DataProcessor()
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
```

---

## Module: src.utils

Utility functions for the system.

### Function: format_currency()

```python
format_currency(amount: float) -> str
```

Format amount as currency string.

**Returns:** formatted currency string (e.g., "$1,234.56")

### Function: get_prediction_message()

```python
get_prediction_message(prediction: int, probability: float = None) -> dict
```

Generate user-friendly prediction message.

**Parameters:**
- prediction: 0 (Not Approved) or 1 (Approved)
- probability: optional prediction probability array

**Returns:** dict with status, message, and confidence

### Function: validate_income_ratio()

```python
validate_income_ratio(
    applicant_income: float,
    coapplicant_income: float,
    loan_amount: float
) -> dict
```

Calculate and validate debt-to-income ratio.

**Returns:** dict with validation status and ratio

**Example:**

```python
from src.utils import format_currency, validate_income_ratio

# Format currency
formatted = format_currency(1234.56)  # "$1,234.56"

# Validate income ratio
ratio = validate_income_ratio(75000, 35000, 180000)
# Returns: {'valid': True, 'ratio': 1.35, 'message': 'Debt-to-Income Ratio: 135%'}
```

---

## Module: src.logger

Logging configuration.

### Function: setup_logger()

```python
setup_logger(name: str) -> logging.Logger
```

Configure and return a logger instance.

**Parameters:** name - logger name (typically __name__)

**Returns:** configured Logger instance

### Variable: logger

Default logger instance configured for the module.

**Example:**

```python
from src.logger import logger

logger.info("Processing loan application")
logger.error("Failed to load model")
```

---

## Streamlit Application (app.py)

Web interface for loan approval prediction.

### Features

- Interactive form for loan application details
- Real-time input validation
- Debt-to-income ratio calculation
- Instant ML prediction with confidence metrics
- Visual result display with color-coded approval status
- Sidebar with application information and tips

### Running

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Training Script (train_model.py)

Model training and evaluation pipeline.

### Functions

- `load_data(file_path)` - Load CSV data
- `preprocess_data(df)` - Handle missing values and encoding
- `train_model(X_train, y_train)` - Train Random Forest classifier
- `evaluate_model(model, X_test, y_test)` - Evaluate model performance
- `save_model(model, save_path)` - Save model to pickle file
- `main()` - Execute complete pipeline

### Running

```bash
python train_model.py
```

---

## Data Format

### Input Features

| Feature | Type | Values | Example |
|---------|------|--------|---------|
| Gender | string | "Male", "Female" | "Male" |
| Married | string | "Yes", "No" | "Yes" |
| Dependents | string | "0", "1", "2", "3+" | "1" |
| Education | string | "Graduate", "Not Graduate" | "Graduate" |
| Self_Employed | string | "Yes", "No" | "No" |
| ApplicantIncome | float | > 0 | 75000 |
| CoapplicantIncome | float | >= 0 | 35000 |
| LoanAmount | float | > 0 | 180000 |
| Loan_Amount_Term | float | > 0 | 360 |
| Credit_History | int | 0 or 1 | 1 |
| Property_Area | string | "Urban", "Rural", "Semiurban" | "Urban" |

### Output

```python
{
    'status': '✅ APPROVED' or '❌ NOT APPROVED',
    'message': 'User-friendly message',
    'confidence': 'Confidence percentage string'
}
```

---

## Error Handling

All modules include comprehensive error handling:

- **InputError**: Invalid input parameters
- **ModelError**: Model loading/prediction failures
- **FileError**: Data file not found or corrupt
- **ProcessingError**: Data processing failures

All errors are logged with full context.

---

## Version & Compatibility

- **Python**: 3.10+
- **scikit-learn**: 1.3.0+
- **pandas**: 2.0.0+
- **streamlit**: 1.28.0+

---

For more information, see README.md and source code documentation.
