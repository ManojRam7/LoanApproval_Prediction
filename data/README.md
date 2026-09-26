# Data

`LoanApprovalPrediction.csv`: 598 loan applications with 11 input fields and the outcome.

| Column | Meaning |
|---|---|
| Loan_ID | Application ID (dropped before training) |
| Gender, Married, Dependents, Education, Self_Employed | Applicant profile |
| ApplicantIncome, CoapplicantIncome | Monthly income |
| LoanAmount, Loan_Amount_Term | Amount requested (thousands) and term (months) |
| Credit_History | 1 if the credit history meets guidelines |
| Property_Area | Urban, Semiurban or Rural |
| Loan_Status | Y approved (411), N rejected (187) |

Missing values (96 cells in total) are filled with the column mean for numeric fields and the mode
for categorical fields.
