import pandas as pd

# Replace 'your_file.dta' with the path to your DTA file
df = pd.read_stata('/g100/home/userexternal/ddurmush/Data_preprocessing/ZA7505_v4-0-0.dta', convert_categoricals=False)

# Now df is a DataFrame containing the data from your DTA file
print(df.head())  # This will print the first few rows of the DataFrame
print(df.columns)
print(len(df))

df = df[df['cntry']==380]
print(len(df))
print(df['reg_iso'].unique)