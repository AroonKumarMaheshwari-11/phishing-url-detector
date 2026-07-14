import pandas as pd

url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv"
df = pd.read_csv(url)

print(df.head())
print(df.info())
print(df.columns.tolist())
print(df.shape)
print(df['phishing'].value_counts())