import pandas as pd

balcon = pd.read_excel(io="./balcon.xlsx")
a = balcon[balcon["cpu"]>=8]
a.groupby(["example_id", "max_layer"]).mean()