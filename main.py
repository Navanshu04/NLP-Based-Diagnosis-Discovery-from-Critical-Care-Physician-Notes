# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
df_chunks = pd.read_csv("C:/Users/Asus/Desktop/NOTEEVENTS.csv", chunksize=100000)

# Get the first chunk
chunk = next(df_chunks)

# Get the first 10 rows of the chunk
first_10_rows = chunk.head(1000)

# Write the first 10 rows to a new file
chunk.to_csv("C:/Users/Asus/Desktop/first_10_rows.csv", index=False)