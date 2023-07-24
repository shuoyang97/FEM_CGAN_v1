import pandas as pd
  
# read contents of csv file
file = pd.read_csv("./dataset/input_data_TOTDISP.csv")
print("\nOriginal file:")
print(file.head)
  
# adding header
headerList = ['fri_mue_data_1', 'fri_mue_data_2','fri_mue_data_3','fri_mue_data_4','fri_mue_data_5','fri_mue_data_6','fri_mue_data_7',
              'Punchforce_data', 'Mesh_Matrize', 'Mesh_Probe', 'Remeshing_Probe', 'material']
  
# converting data frame to csv
file.to_csv("./dataset/input_data_TOTDISP.csv", header=headerList, index=False)
file2 = pd.read_csv("./dataset/input_data_TOTDISP.csv")
print('\nModified file:')
print(file2.head)