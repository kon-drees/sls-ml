import pandas as pd
import os



def correct():
    csv_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_initial_argumentation_frameworks'
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(csv_folder, csv_file), header=None)
        df.columns = ['Argument', 'Label']
        df.sort_values('Argument', inplace=True)
        df.to_csv(os.path.join(csv_folder, csv_file), index=False)



if __name__ == '__main__':
    # Set the paths to your folders
    argumentation_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/argumentation_frameworks'
    processed_label_folder = '/Users/konraddrees/Documents/GitHub/sls-ml/files/ml_label_argumentation_frameworks'

    # Get the list of files in each folder
    argumentation_files = os.listdir(argumentation_folder)
    label_files = os.listdir(processed_label_folder)

    # Strip the extensions from the file names
    argumentation_files = [os.path.splitext(file)[0] for file in argumentation_files]
    label_files = [os.path.splitext(file)[0].replace("_labels", "") for file in
                   label_files]  # Also remove the "_label" part

    # Find argumentation files with no corresponding label
    missing_labels = list(set(argumentation_files) - set(label_files))

    # Print the argumentation files with missing labels
    for file in missing_labels:
        print(f"The file '{file}' in the argumentation folder does not have a corresponding label in the label folder.")

