import os
import nibabel as nib
import numpy as np
import datasets as ds
import arguments as ag
import utils.DimensionUtils as du


class BrainDatasetParams(ds.DatasetParams):
    def __init__(self, root_folder: str, samples: [ds.PDSample], txt_indices: [int], down_sample_ratio: int = 1):
        super().__init__(root_folder, samples)
        self.txt_indices = txt_indices
        self.down_sample_ratio = down_sample_ratio


class BrainDataset(ds.WholeDataset):
    def __init__(self, params: BrainDatasetParams):
        super().__init__(params)
        self.params = params

    def __getitem__(self, index: int) -> ([[[float]]], [float], [[[float]]], [[[float]]]):
        sample = self.params.samples[index]
        event_path = ag.BDArguments.event_parts[sample.event_index]

        file_path = os.path.join(self.params.root_folder, event_path, sample.No)
        paths = []
        for filename_format in ag.BDArguments.NII_FILE_NAMES:
            path = os.path.join(file_path, filename_format)
            path = path.format(sample.No)
            paths.append(path)

        images = []
        for i in range(len(ag.BDArguments.NII_FILE_NAMES)):
            if i not in self.params.txt_indices:
                continue
            path = paths[i]
            image = nib.load(path).get_fdata()
            if self.params.down_sample_ratio > 1:
                image = du.down_sample_3D(image, self.params.down_sample_ratio)
            images.append(image)

        images = np.array(images, dtype=np.float32)
        return images, sample.label
