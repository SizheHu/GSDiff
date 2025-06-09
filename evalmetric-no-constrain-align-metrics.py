import os
import numpy as np
import torch
from tqdm import *

'''This is used to calculate the alignment loss and false edge ratio for 3000 unconstrained inference results (corresponding to the ablation experiment part in the paper)'''

batch_size = 1
device = 'cpu'

# Initialize a dictionary to store all FID and KID values ​​and alignment losses
metrics_dict = {}

# Traverse all folders in the current directory
for folder in os.listdir('./T'):
    if os.path.isdir('./T/' + folder) and '-' in folder:
        if len(folder.split('-')) == 2 and folder.split('-')[0].isupper() and folder.split('-')[1].isdigit():
            # The average alignment loss of the 3000 samples in this experiment number group
            avg_align_loss = 0
            # Traversing the vr4stat_i.npy file
            for i in tqdm(range(3000)):
                file_name = f"vr4stat_{i}.npy"
                file_path = os.path.join('./T/' + folder, file_name)

                # Check if a file exists
                if os.path.isfile(file_path):
                    # Loading numpy arrays
                    data = np.load(file_path, allow_pickle=True).item()
                    # Extract the value corresponding to the 'output_points_test' key and convert it to (1, 53, 2) (the value after that is 99999)
                    output_points_test = torch.tensor(np.array([list(tupl)[:2] for tupl in data['output_points_test']])[None, :, :])

                    pred_xstart_cs = torch.zeros((1, 53, 10))
                    pred_xstart_cs[:, 0:output_points_test.shape[1], 0:2] = output_points_test
                    pred_xstart_cs[:, output_points_test.shape[1]:, 9:10] = 1

                    # Filter the corresponding rows based on whether the last column is 0
                    mask = pred_xstart_cs[:, :, -1] == 0

                    # Select only the first two columns (x and y coordinates) of the rows where the last column is 0
                    x_coords = pred_xstart_cs[:, :, 0] * mask
                    y_coords = pred_xstart_cs[:, :, 1] * mask

                    # Set the False values ​​in mask to infinity to exclude these values ​​when calculating distance
                    inf_mask = torch.where(mask, 0, float('inf'))
                    x_coords += inf_mask
                    y_coords += inf_mask

                    # Compute the L1 distance matrix of the x-coordinate
                    x_coords_uns = x_coords.unsqueeze(2)
                    distance_matrix_x = torch.abs(x_coords_uns - x_coords_uns.transpose(1, 2))

                    # Compute the L1 distance matrix of the y coordinates
                    y_coords_uns = y_coords.unsqueeze(2)
                    distance_matrix_y = torch.abs(y_coords_uns - y_coords_uns.transpose(1, 2))

                    # Set infinity and NaN values ​​to 99999
                    distance_matrix_x[torch.isinf(distance_matrix_x) | torch.isnan(distance_matrix_x)] = 99999
                    distance_matrix_y[torch.isinf(distance_matrix_y) | torch.isnan(distance_matrix_y)] = 99999
                    # Diagonal lines are filled with mask 99999
                    distance_matrix_x[
                        (torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_x.size(0), 53, 53)] = 99999
                    distance_matrix_y[
                        (torch.eye(53).unsqueeze(0) == 1).to(device).expand(distance_matrix_y.size(0), 53, 53)] = 99999
                    # print(distance_matrix_x)
                    # print(distance_matrix_y)
                    # Calculate the minimum of all x, y axial distances for each node i
                    min_values_x, _ = torch.min(distance_matrix_x, dim=2)
                    min_values_y, _ = torch.min(distance_matrix_y, dim=2)
                    # print(min_values_x.shape) # (bs, 53)
                    # print(min_values_y.shape) # (bs, 53)
                    # Find the node i with the smallest distance (106) to other nodes (53) in all directions (2)
                    min_values = torch.stack((min_values_x, min_values_y), dim=2)
                    min_values, _ = torch.min(min_values, dim=2)
                    # print(min_values.shape) # (bs, 53) contains 99999, which means the node is a padding node
                    # For each sample, put the non-99999 elements on g=-2log(1-(0.5-eps)x)

                    # Apply the function -2log(1-0.5x) to the non-99999 elements
                    # Note: Make sure the value you input to log is positive, i.e. 1-0.5x > 0
                    # 99999 will directly become 0 here, no effect
                    # Multiply by the time weight
                    masked_tensor = torch.where(min_values != 99999,
                                                min_values,
                                                torch.tensor(0.0, dtype=min_values.dtype, device=device)).sum(1)
                    # print(masked_tensor.shape)
                    # print(masked_tensor)

                    # Sum the results
                    avg_align_loss += masked_tensor.sum()

            avg_align_loss /= 3000

            # Get the number from the folder name
            group, number = folder.split('-')
            # Add data to the dictionary
            if number not in metrics_dict:
                metrics_dict[number] = {'Alg': []}
            metrics_dict[number]['Alg'].append(avg_align_loss)
# Output the average value of each number
print(f"{'Number':<10}{'Alg':<20}")
for number, values in sorted(metrics_dict.items(), key=lambda x: int(x[0])):
    avg_alg = np.mean(np.array(values['Alg']))
    print(f"{number:<10}{avg_alg:<20}")





