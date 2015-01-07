#Load the data from the csv file using pandas
import pandas as pd
df = pd.read_csv('train.tsv', header=0, delimiter='\t')
print df.count()