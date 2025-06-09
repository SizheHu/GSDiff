import os
import numpy as np

'''This is used to calculate the FID and KID (located in 'test_metrics.npy') calculated from the results of any inference under any constraints'''

# Initialize a dictionary to store all FID and KID values
metrics_dict = {}

# Traverse all folders in the current directory
for folder in os.listdir('./T'):
    if os.path.isdir('./T/' + folder) and '-' in folder:
        if len(folder.split('-')) == 2 and folder.split('-')[0].isupper() and folder.split('-')[1].isdigit():
            # Construct the full path to the test_metrics.npy file
            file_path = os.path.join('./T/' + folder, 'test_metrics.npy')
            # Check if a file exists
            if os.path.isfile(file_path):
                # Loading numpy arrays
                data = np.load(file_path, allow_pickle=True)[0]
                # Get FID and KID values
                fid_value = float(data[1])
                kid_value = float(data[2])
                # Get the number from the folder name
                group, number = folder.split('-')
                # Add data to the dictionary
                if number not in metrics_dict:
                    metrics_dict[number] = {'FID': [], 'KID': []}
                metrics_dict[number]['FID'].append(fid_value)
                metrics_dict[number]['KID'].append(kid_value)

# Output the average FID and KID for each ID
print(f"{'Number':<10}{'FID Average':<20}{'KID Average':<20}")
for number, values in sorted(metrics_dict.items(), key=lambda x: int(x[0])):
    # print(values['FID'], type(values['FID']))
    avg_fid = np.mean(np.array(values['FID']))
    avg_kid = np.mean(np.array(values['KID']))
    print(f"{number:<10}{avg_fid:<20}{avg_kid:<20}")
