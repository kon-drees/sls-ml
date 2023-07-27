import pandas as pd
import os


if __name__ == '__main__':
    csv_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_initial_argumentation_frameworks'
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    for csv_file in csv_files:

        df = pd.read_csv(os.path.join(csv_folder, csv_file), header=None)
        df.columns = ['Argument', 'Label']
        df.sort_values('Argument', inplace=True)
        df.to_csv(os.path.join(csv_folder, csv_file), index=False)
