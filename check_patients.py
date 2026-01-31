import pandas as pd

df = pd.read_excel("data/excel/patient_clinical_data.xlsx")
print(df.columns.tolist())
