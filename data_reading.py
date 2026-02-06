import zipfile, os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.io import loadmat

dataset = 3
if dataset == 1:
    # Path to the .zip file
    zip_path = './data/raw/our_data.zip'

    # Path to the .pkl file inside the .zip
    folder_path = 'our_data/'
    # Initialize a global list to store all datasets
    all_datasets = []

    with zipfile.ZipFile(zip_path, 'r') as archive:
        # List all files in the specified folder
        file_list = [f for f in archive.namelist() if f.startswith(folder_path) and f.endswith('.pkl')] 
        
        # Iterate through each .pkl file
        for pkl_file_path in file_list:
            with archive.open(pkl_file_path) as pkl_file:
                # Load the .pkl file using pickle
                data = pkl.load(pkl_file)
                # Extract the data for the current battery
                processing_bat = pkl_file_path.split('/')[1].split('.')[0] 
                data = data[processing_bat]
                
                # Initialize an empty list to store the combined data for the current file
                combined_data = []

                # Iterate through each cycle in the dataset
                for cycle in data['rul'].keys():
                    print(f'Processing {processing_bat} battery at cycle {cycle}')
                    # Extract the dataset for the current cycle
                    cycle_data = data['data'][cycle]
                    
                    # Add the 'rul' and 'dq' values as new columns
                    cycle_data['rul'] = data['rul'][cycle]
                    cycle_data['dq'] = data['dq'][cycle]
                    
                    # Append the modified dataset to the combined_data list
                    combined_data.append(cycle_data)

                # Concatenate all the dataframes in the combined_data list into a single dataframe
                final_dataset = pd.concat(combined_data, ignore_index=True)
                final_dataset['bat_name'] = processing_bat
                
                # Append the final dataset for the current file to the global list
                all_datasets.append(final_dataset)

    # Concatenate all datasets into a single dataframe
    full_dataset = pd.concat(all_datasets, ignore_index=True)
    full_dataset.to_pickle('./data/processed/MENDELEY/DataSet_mendeley.pkl')

elif dataset == 2: 
    # Path to the .zip file
    zip_path = './data/raw/Battery Dataset.zip'

    # Path to the folder inside the .zip
    folder_path = 'Battery Dataset/'
    output_folder = './data/raw/extracted_pdfs/'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize a global list to store all datasets
    all_datasets = []

    with zipfile.ZipFile(zip_path, 'r') as archive:
        # List all .mat files in the specified folder and its subfolders
        file_list = [f for f in archive.namelist() if f.startswith(folder_path) and f.endswith('.mat')]
        pdf_list = [f for f in archive.namelist() if f.startswith(folder_path) and f.endswith('.pdf')]

        # Extract and save each .pdf file
        for pdf_file_path in pdf_list:
            # Extract the file content
            with archive.open(pdf_file_path) as pdf_file:
                # Define the output file path
                output_file_path = os.path.join(output_folder, os.path.basename(pdf_file_path))
                
                # Save the file to the output folder
                with open(output_file_path, 'wb') as output_file:
                    output_file.write(pdf_file.read())
        # Iterate through each .mat file
        for mat_file_path in file_list:
            print(f'Processing the file {mat_file_path}') # in the folder: {}')
            with archive.open(mat_file_path) as mat_file:
                # Load the .mat file
                mat_data = loadmat(mat_file)
                
                # Extract the 'data' field
                data = mat_data['data']
                
                # Check if 'data' is a structured array
                if isinstance(data, np.ndarray):
                    # Convert the structured array into a dictionary
                    data_dict = {field: data[field][0, 0] for field in data.dtype.names}
                    
                    # Handle fields with single values (e.g., 'description')
                    if len(data_dict['description']) == 1:
                        data_dict['description'] = [data_dict['description'][0]] * len(data_dict['system_time'])
                        data_dict['header'] = [mat_data['__header__'][0]] * len(data_dict['system_time'])
                        data_dict['version'] = [mat_data['__version__'][0]] * len(data_dict['system_time'])
                        if len(mat_data['__globals__']) > 0:
                            data_dict['globals'] = [mat_data['__globals__']] * len(data_dict['system_time'])
                    
                    # Flatten all fields to ensure they are 1-dimensional
                    for key in data_dict.keys():
                        data_dict[key] = np.array(data_dict[key]).ravel()
                    
                    # Create a DataFrame from the dictionary
                    df = pd.DataFrame(data_dict)
                    
                    # Add a column to identify the source file
                    df['source_file'] = mat_file_path
                    
                    # Append the DataFrame to the global list
                    all_datasets.append(df)

    # Concatenate all DataFrames into a single DataFrame
    final_dataset = pd.concat(all_datasets, ignore_index=True)
    final_dataset.to_pickle('./data/processed/PROGNOSIS/DataSet_prognosis.pkl')
else:
    data_file = './data/processed/NASA/DataSetRulEstimation_all_new.csv'
    df_battery = pd.read_csv(data_file, sep=',')
    df_battery.to_pickle('./data/processed/NASA/DataSetRulEstimation_all_new.pkl')
