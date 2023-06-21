import torch
import numpy as np
import xarray as xr

def temporal_dataloader(dataset, feature_names, target_name, lat_size, lon_size, time_size, sequence_length, shuffle=True, num_workers=0):
    # Create index arrays for each dimension
    batch_size = [lat_size, lon_size, time_size] #width of batch along each dimension
    total_lat_size = dataset.sizes['latitude']
    total_lon_size = dataset.sizes['longitude']
    total_time_size = dataset.sizes['time']
    indices_dim1 = np.arange(total_lat_size) #array with len of the first dimension
    indices_dim2 = np.arange(total_lon_size)
    indices_dim3 = np.arange(total_time_size)

    if shuffle:
        np.random.shuffle(indices_dim1) #do not care about spatial relationships yet
        np.random.shuffle(indices_dim2)

    # Divide the indices of each dimension into batches
    batch_indices_dim1 = [indices_dim1[i:i + batch_size[0]] for i in range(0, total_lat_size, batch_size[0])]
    batch_indices_dim2 = [indices_dim2[i:i + batch_size[1]] for i in range(0, total_lon_size, batch_size[1])]
    batch_indices_dim3 = [indices_dim3[i:i + batch_size[2]] for i in range(0, total_time_size, batch_size[2])]

    if shuffle: #shuffle batch time indices while maintaining temporal relationships within each batch
        np.random.shuffle(batch_indices_dim3)

    # Define a generator function to yield batches
    def generator():
        #feature output tensor needs dimensions (batch_size, sequence_length, len(feature_names)) where batch_size=lat_size*lon_size. ignore padding for convolutions for now
        #target output tensor needs dimensions (batch_size, 1) because we only have one target variable to predict
        for sample_indices_d1 in batch_indices_dim1: #each batch_index is the list of individual indices within that batch
            for sample_indices_d2 in batch_indices_dim2: #lon
                for sample_indices_d3 in batch_indices_dim3: #time
                    for i, _ in enumerate(sample_indices_d3[0:time_size-sequence_length-1]): #create time series, sequence_length+1 is number of timesteps looked at for feature and targets combined
                        
                        #has shape (lat_size, lon_size, sequence_length, len(feature_names))
                        features = dataset.isel(latitude=sample_indices_d1,longitude=sample_indices_d2,time=sample_indices_d3[i:i+sequence_length])[feature_names] #shape latsize, lonsize, sequence_length, featuresize
                        features = features.stack(samples=("latitude", "longitude")) #combine dimensions. now it is batch size, time, features where batch size is lat*lon
                        #print(features.to_array().values.shape) 28 by 64 by 16
                        features_tensor = torch.from_numpy(features.to_array().values)
                        
                        #has shape (lat_size, lon_size, 1, 1) because one time index and one target variable
                        target = dataset.isel(latitude=sample_indices_d1,longitude=sample_indices_d2,time=sample_indices_d3[i+sequence_length])[target_name] #the value the next timestep after, shape latsize, lonsize, timesize, targetsize
                        print(target)
                        print(target.to_array().values)
                        target = target.stack(samples=("latitude", "longitude")) #combine dimensions again
                        target_tensor = torch.from_numpy(target.to_array().values)[:,0,:] #remove the time dimension because only one time index for one target
                        yield features_tensor, target_tensor 
                    

    # Return an iterator that yields batches
    return iter(generator())
