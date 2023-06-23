import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset, DataLoader

class TemporalDataset(Dataset):
    def __init__(self, dataset, feature_names, target_name, batch_indices_dim1, batch_indices_dim2, batch_indices_dim3, sequence_length):
        self.dataset = dataset
        self.feature_names = feature_names
        self.target_name = target_name
        self.batch_indices_dim1 = batch_indices_dim1
        self.batch_indices_dim2 = batch_indices_dim2
        self.batch_indices_dim3 = batch_indices_dim3
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.batch_indices_dim1) * len(self.batch_indices_dim2) * len(self.batch_indices_dim3)

    def __getitem__(self, index):
        batch_index_dim1 = index // (len(self.batch_indices_dim2) * len(self.batch_indices_dim3))
        batch_index_dim2 = (index % (len(self.batch_indices_dim2) * len(self.batch_indices_dim3))) // len(self.batch_indices_dim3)
        batch_index_dim3 = (index % (len(self.batch_indices_dim2) * len(self.batch_indices_dim3))) % len(self.batch_indices_dim3)

        sample_indices_d1 = self.batch_indices_dim1[batch_index_dim1]
        sample_indices_d2 = self.batch_indices_dim2[batch_index_dim2]
        sample_indices_d3 = self.batch_indices_dim3[batch_index_dim3]

        features = self.dataset.isel(latitude=sample_indices_d1, longitude=sample_indices_d2, time=sample_indices_d3[:self.sequence_length])
        features = features.stack(samples=("latitude", "longitude")).transpose('samples', 'time')
        features_np = np.moveaxis(features.to_array().values, 0, -1)
        features_tensor = torch.from_numpy(features_np)

        target = self.dataset.isel(latitude=sample_indices_d1, longitude=sample_indices_d2, time=sample_indices_d3[self.sequence_length])[self.target_name]
        target = target.stack(samples=("latitude", "longitude")).transpose('samples')
        target_np = np.moveaxis(target.to_array().values, 0, -1)
        target_tensor = torch.from_numpy(target_np)

        return features_tensor, target_tensor

def temporal_dataloader(dataset, feature_names, target_name, lat_size, lon_size, time_size, sequence_length, shuffle=True, num_workers=0, batch_size=1):
    total_lat_size = dataset.sizes['latitude']
    total_lon_size = dataset.sizes['longitude']
    total_time_size = dataset.sizes['time']
    indices_dim1 = np.arange(total_lat_size)
    indices_dim2 = np.arange(total_lon_size)
    indices_dim3 = np.arange(total_time_size)

    if shuffle:
        np.random.shuffle(indices_dim1)
        np.random.shuffle(indices_dim2)
        np.random.shuffle(indices_dim3)

    batch_indices_dim1 = [indices_dim1[i:i + lat_size] for i in range(0, total_lat_size, lat_size)]
    batch_indices_dim2 = [indices_dim2[i:i + lon_size] for i in range(0, total_lon_size, lon_size)]
    batch_indices_dim3 = [indices_dim3[i:i + time_size] for i in range(0, total_time_size, time_size)]

    dataset = TemporalDataset(dataset, feature_names, target_name, batch_indices_dim1, batch_indices_dim2, batch_indices_dim3, sequence_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader
