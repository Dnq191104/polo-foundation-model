import pandas as pd
df = pd.read_parquet("artifacts/step7/pairs.parquet")
print(df.head(10))
