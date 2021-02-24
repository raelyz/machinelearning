import numpy as np
import pandas as pd

# Do the following:

# Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason. Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

# Output the following:

# the entire DataFrame
# the value in the cell of row #1 of the Eleanor column
# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.

my_data = np.random.randint(low=0, high=101, size=(3, 4))

my_column_names = ['Eleanor', 'Chidi', 'Tahini', 'Jason']
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

my_dataframe['Janet'] = my_dataframe['Tahini']+my_dataframe['Jason']
print(my_dataframe['Eleanor'][1])
print(my_dataframe)
