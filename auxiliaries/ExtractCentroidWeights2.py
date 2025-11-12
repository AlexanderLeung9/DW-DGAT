# -*- coding: utf-8 -*-
import os
import nibabel as nib
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import platform
import sys
import random

sys.path.append("..")
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print("using device:", device)
cudnn.enabled = True


def set_random(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_data(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data


def process_data(path_root_dir, output_root_dir, path_template, generate_OR_delete: bool):
    data_types = [
        "T1\\co_{0}_swap_bet_crop_resample_2MNI152.nii.gz",
        "standard_space\\{0}_06LDHs_4normalize_to_target_2mm.nii.gz",
        "standard_space\\{0}_07LDHk_4normalize_to_target_2mm.nii.gz",
        "standard_space\\{0}_FA_4normalize_to_target_2mm.nii.gz",
        "standard_space\\{0}_L1_4normalize_to_target_2mm.nii.gz",
        "standard_space\\{0}_L23m_4normalize_to_target_2mm.nii.gz",
        "standard_space\\{0}_MD_4normalize_to_target_2mm.nii.gz",
    ]
    months = ["0m", "12m", "24m"]

    system = platform.system()
    if system == "Linux":
        data_types2 = []
        for data_type in data_types:
            data_type2 = data_type.replace('\\', '/')
            data_types2.append(data_type2)
        data_types = data_types2

        years = months
    else:
        # years = [os.path.join(month, "DTI_Results_GOOD") for month in months]
        years = months

    template = read_data(path_template)
    print(f"template.shape={template.shape}")
    template = torch.tensor(template, dtype=torch.float).to(device)
    os.makedirs(output_root_dir, exist_ok=True)

    print("Calculating total count...")
    total_count = 0
    for year in years:
        dti_path = os.path.join(path_root_dir, year)
        if os.path.isdir(dti_path):
            for sample_id in os.listdir(dti_path):
                sample_path = os.path.join(dti_path, sample_id)
                if os.path.isdir(sample_path) and sample_id.isdigit() and len(sample_id) == 6:
                    for data_type in data_types:
                        data_type_path = os.path.join(sample_path, data_type.format(sample_id))
                        if os.path.exists(data_type_path):
                            total_count += 1

    count = 0
    cubic_diameter = np.sqrt(91 ** 2 + 109 ** 2 + 91 ** 2)

    for year in years:
        dti_path = os.path.join(path_root_dir, year)
        if os.path.isdir(dti_path):
            for sample_id in os.listdir(dti_path):
                sample_path = os.path.join(dti_path, sample_id)
                if os.path.isdir(sample_path) and sample_id.isdigit() and len(sample_id) == 6:
                    for data_type in data_types:
                        data_type2 = data_type.format(sample_id)
                        data_type_path = os.path.join(sample_path, data_type2)
                        if os.path.exists(data_type_path):
                            count += 1
                            action = "Generating" if not generate_OR_delete else "Deleting"
                            print(f"{action} files with {data_type_path} ({count}/{total_count})")

                            output_file_path = os.path.join(output_root_dir, year, sample_id, "ROI")
                            os.makedirs(output_file_path, exist_ok=True)

                            if system == "Windows":
                                two_parts = data_type2.split('\\')
                            else:
                                two_parts = data_type2.split('/')
                            if two_parts[0] == "T1":
                                metric = "T1"
                            else:
                                metric = two_parts[1].split('_')[1]
                            max_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-MaxValue_AAL-90-2mm.txt")
                            avg_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-AvgValue_AAL-90-2mm.txt")
                            centroid_weight_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-CentroidWeight_AAL-90-2mm.txt")
                            centroid_position_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-CentroidPosition_AAL-90-2mm.txt")
                            centroid_length_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-CentroidLength_AAL-90-2mm.txt")
                            centroid_cos_angles_file_path = os.path.join(output_file_path, f"{sample_id}_{metric}-CentroidCosAgl_AAL-90-2mm.txt")

                            if generate_OR_delete:
                                if os.path.isfile(max_file_path):
                                    os.remove(max_file_path)
                                if os.path.isfile(avg_file_path):
                                    os.remove(avg_file_path)
                                if os.path.isfile(centroid_weight_file_path):
                                    os.remove(centroid_weight_file_path)
                                if os.path.isfile(centroid_position_file_path):
                                    os.remove(centroid_position_file_path)
                                if os.path.isfile(centroid_length_file_path):
                                    os.remove(centroid_length_file_path)
                                if os.path.isfile(centroid_cos_angles_file_path):
                                    os.remove(centroid_cos_angles_file_path)
                                continue

                            avg_values = []
                            max_values = []
                            centroid_positions = []
                            centroid_weights = []
                            centroid_lengths = []
                            centroid_cos_angles = []
                            data = read_data(data_type_path)
                            data = torch.tensor(data, dtype=torch.float).to(device)

                            for roi_number in range(1, 91):
                                mask = template == roi_number

                                roi_data = data[mask]
                                avg_value = torch.mean(roi_data)
                                max_value = torch.max(roi_data)

                                masked_data = data * mask
                                x_coordinates = torch.arange(template.shape[0], dtype=torch.float32).to(device)
                                y_coordinates = torch.arange(template.shape[1], dtype=torch.float32).to(device)
                                z_coordinates = torch.arange(template.shape[2], dtype=torch.float32).to(device)

                                t_total_weight = masked_data.sum()
                                if t_total_weight == 0:
                                    print(f"{data_type2} ROI{roi_number}: t_total_weight == 0")
                                    data2 = torch.ones_like(masked_data, dtype=torch.float32).to(device)
                                    masked_data = data2 * mask
                                    total_weight = masked_data.sum()
                                else:
                                    total_weight = t_total_weight

                                x_weighted = (masked_data * x_coordinates.view(-1, 1, 1)).sum()
                                y_weighted = (masked_data * y_coordinates.view(1, -1, 1)).sum()
                                z_weighted = (masked_data * z_coordinates.view(1, 1, -1)).sum()

                                centroid_x = x_weighted / total_weight
                                centroid_y = y_weighted / total_weight
                                centroid_z = z_weighted / total_weight
                                centroid_x2 = centroid_x.item()
                                centroid_y2 = centroid_y.item()
                                centroid_z2 = centroid_z.item()
                                centroid_pos = (centroid_x2, centroid_y2, centroid_z2)

                                x = torch.round(centroid_x).type(torch.long)
                                y = torch.round(centroid_y).type(torch.long)
                                z = torch.round(centroid_z).type(torch.long)

                                if t_total_weight == 0:
                                    centroid_weight = torch.tensor(0, dtype=torch.float32)
                                else:
                                    centroid_weight = masked_data[x, y, z]

                                if metric == "T1" and max_value != 0:
                                    avg_value /= max_value
                                    centroid_weight /= max_value

                                avg_value2 = avg_value.item()
                                max_value2 = max_value.item()
                                centroid_weight2 = centroid_weight.item()

                                avg_values.append(avg_value2)
                                max_values.append(max_value2)
                                centroid_positions.append(centroid_pos)
                                centroid_weights.append(centroid_weight2)

                                centroid_length = torch.sqrt(centroid_x ** 2 + centroid_y ** 2 + centroid_z ** 2)
                                cos_angle_x = centroid_x / centroid_length
                                cos_angle_y = centroid_y / centroid_length
                                cos_angle_z = centroid_z / centroid_length
                                # Normalize vector lengths.
                                centroid_length /= cubic_diameter
                                centroid_lengths.append(centroid_length.item())
                                centroid_cos_angles.append((cos_angle_x.item(), cos_angle_y.item(), cos_angle_z.item()))

                            if metric == "T1":
                                max_values = np.array(max_values)
                                tMax_value = np.max(max_values)
                                if tMax_value != 0:
                                    max_values /= tMax_value

                            with (open(max_file_path, 'w') as max_file, open(avg_file_path, 'w') as avg_file, open(centroid_position_file_path, 'w') as centroid_pos_file, open(centroid_weight_file_path, 'w') as centroid_weight_file, open(centroid_length_file_path, 'w') as centroid_length_file, open(centroid_cos_angles_file_path, 'w') as centroid_cos_angles_file):
                                max_values2 = "\n".join(map(str, max_values))
                                max_file.write(max_values2)

                                avg_values2 = "\n".join(map(str, avg_values))
                                avg_file.write(avg_values2)

                                centroid_positions2 = "\n".join(["\t".join(map(str, coord)) for coord in centroid_positions])
                                centroid_pos_file.write(centroid_positions2)

                                centroid_weights2 = "\n".join(map(str, centroid_weights))
                                centroid_weight_file.write(centroid_weights2)

                                centroid_lengths2 = "\n".join(map(str, centroid_lengths))
                                centroid_length_file.write(centroid_lengths2)

                                centroid_cos_angles2 = "\n".join(["\t".join(map(str, cos_angles)) for cos_angles in centroid_cos_angles])
                                centroid_cos_angles_file.write(centroid_cos_angles2)


def main():
    parser = argparse.ArgumentParser(description="Extract centroid weights")
    parser.add_argument("-data_dir", help="Path to the root directory", default=r"E:\BDdata\PD_middle_data\MRI_DTI")
    parser.add_argument("-output_dir", help="Path to the output directory", default=r"E:\BDdata\PD_middle_data\MRI_DTI")
    parser.add_argument("-template_path", help="Path to the template file", default=r"./documents/AAL_Contract_90_2MM.nii.gz")
    parser.add_argument("-generate_OR_delete", type=bool, help="Generate or delete files from or to the output directory", default=False)
    args = parser.parse_args()

    set_random(231)
    process_data(args.data_dir, args.output_dir, args.template_path, args.generate_OR_delete)


if __name__ == "__main__":
    main()
