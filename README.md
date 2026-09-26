# Loan Approval Prediction

Predicts whether a loan application will be approved from the applicant's profile, income, loan
amount, term, credit history and property area. Four classifiers were compared in the notebook; the
random forest is served in a Streamlit app that also shows the debt-to-income picture and the
model's confidence for each prediction.

**Live app:** https://loanapprovalprediction-app.streamlit.app

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## Results

598 applications (411 approved, 187 rejected), 60/40 train/test split (358 / 240):

| Model | Test accuracy |
|---|---|
| **Random forest** (7 trees, entropy) | **82.5%** |
| Logistic regression | 80.8% |
| SVC | 69.2% |
| K-nearest neighbours | 63.8% |

The random forest reaches 98% on the training set against 82.5% on test, so it overfits; logistic
regression is within two points and would be the safer choice if the model were refit on new data.

## Approach

`notebooks/Loan_Approval_Analysis.ipynb`:

1. Drop `Loan_ID`, label-encode the binary categorical fields, fill missing values (mean for
   numbers, mode for categories).
2. Check value counts per column and the correlation heatmap; credit history is the strongest
   single signal for approval.
3. Train random forest, KNN, SVC and logistic regression on the same split and compare accuracy.

The same steps are packaged for the app in `src/`: `data_processor.py` prepares inputs,
`model_manager.py` loads the model (training it on first launch if the file is missing), and
`utils.py` formats the result and checks the income-to-loan ratio.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_model.py        # trains and saves models/random_forest_model.pkl
streamlit run app.py
```

## Project structure

```text
app.py                        Streamlit app
train_model.py                training script
src/
    config.py                 paths, model settings, split
    data_processor.py         input encoding and validation
    model_manager.py          model loading and prediction
    utils.py                  formatting and ratio checks
    logger.py                 logging
notebooks/                    analysis and model comparison
data/                         dataset and data dictionary
models/                       trained model (generated)
```

## Limitations

A small dataset (598 rows), a single split and accuracy as the only metric. Next steps would be
stratified cross-validation, precision and recall for the rejected class, and a check for bias
across gender and marital status before any real use.
