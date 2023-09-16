import pandas as pd

csv_file = 'extract.csv'

df = pd.read_csv(csv_file)

latex_table = df.to_latex(index=False,escape=False)
print(latex_table)