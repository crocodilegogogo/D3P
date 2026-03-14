import random as rd
import math
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class D3PSamplerBase(Sampler):
    def __init__(self, dataset, indices, batch_size=None,
                 padding_data=False, shuffle=True):
        super().__init__(data_source=None)

        if padding_data and batch_size is None:
            raise RuntimeError(f'If you want to pad the data (padding_data) of each batch, '
                            f'you need to provide the batch_size parameter')

        self.dataset = dataset
        self.length = len(dataset)
        self.indices = indices
        self.batch_size = batch_size
        self.padding_data = padding_data
        self.shuffle = shuffle
        self.num_samples = self.calculate_num_samples(self.length)

    def calculate_num_samples(self, data_size):
        if self.padding_data:
            num_batch = int(math.ceil(data_size / self.batch_size))
            return num_batch * self.batch_size
        return data_size

    def update_indices(self, indices):
        self.indices = indices
        self.num_samples = self.calculate_num_samples(self.length)

    def shuffle_and_slice(self, datalist):
        if self.shuffle:
            rd.shuffle(datalist)

        data_size = len(datalist)
        if data_size > 0 and self.padding_data:
            repeat_times = int(math.ceil(self.num_samples / data_size))
            datalist = (datalist * repeat_times)[:self.num_samples]

        return iter(datalist)

    def __iter__(self):
        datalist = list(range(self.length))
        return self.shuffle_and_slice(datalist)

    def __len__(self):
        return self.num_samples


class D3PSamplerRemain(D3PSamplerBase):
    def __init__(self, dataset, indices, batch_size=None,
                 padding_data=False, shuffle=True):
        super().__init__(dataset, indices, batch_size, padding_data, shuffle)
        self.num_samples = self.calculate_num_samples(self.length - len(indices))

    def update_indices(self, indices):
        self.indices = indices
        self.num_samples = self.calculate_num_samples(self.length - len(indices))

    def __iter__(self):
        dataset_idx = list(range(self.length))
        indices_set = set(self.indices)
        datalist = [item for item in dataset_idx if item not in indices_set]
        return self.shuffle_and_slice(datalist)


class D3PSamplerPrune(D3PSamplerBase):
    def __init__(self, dataset, indices, batch_size=None,
                 padding_data=False, shuffle=True):
        super().__init__(dataset, indices, batch_size, padding_data, shuffle)
        self.num_samples = self.calculate_num_samples(len(indices))

    def update_indices(self, indices):
        self.indices = indices
        self.num_samples = self.calculate_num_samples(len(indices))

    def __iter__(self):
        datalist = self.indices
        return self.shuffle_and_slice(datalist)


class D3PDistributedSamplerBase(DistributedSampler):
    def __init__(self, dataset, indices, batch_size=None, padding_data=False,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        # The D3P algorithm needs to keep "drop_last=False"
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=False)

        if padding_data and batch_size is None:
            raise RuntimeError(f'If you want to pad the data (padding_data) of each batch, '
                            f'you need to provide the batch_size parameter')

        self.length = len(self.dataset)
        self.indices = indices
        self.batch_size = batch_size
        self.padding_data = padding_data
        self.num_replicas = num_replicas
        self.num_samples = self.calculate_num_samples(self.length)
        self.total_size = self.num_samples * self.num_replicas


    def calculate_num_samples(self, data_size):
        num_samples = int(math.ceil(data_size / self.num_replicas))
        if self.padding_data:
            num_batch = int(math.ceil(num_samples / self.batch_size))
            num_samples = num_batch * self.batch_size
        return num_samples

    def update_indices(self, indices):
        self.indices = indices
        self.num_samples = self.calculate_num_samples(self.length)
        self.total_size = self.num_samples * self.num_replicas

    def shuffle_and_slice(self, datalist):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            permuted_indices = torch.randperm(len(datalist), generator=g).tolist()
            datalist = [datalist[i] for i in permuted_indices]

        total_size = len(datalist)
        if total_size > 0:
            if total_size < self.total_size:
                repeat_times = int(math.ceil(self.total_size / total_size))
                datalist = (datalist * repeat_times)[:self.total_size]
            else:
                datalist = datalist[:self.total_size]

            start_index = self.rank * self.num_samples
            end_index = start_index + self.num_samples
            data_set = datalist[start_index:end_index]
            return iter(data_set)
        else:
            return iter(datalist)


    def __iter__(self):
        datalist = list(range(self.length))
        return self.shuffle_and_slice(datalist)


    def __len__(self):
        return self.num_samples


class D3PDistributedSamplerRemain(D3PDistributedSamplerBase):

    def __init__(self, dataset, indices, batch_size=None, padding_data=False,
                 num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, indices, batch_size, padding_data,
                         num_replicas, rank, shuffle, seed)
        self.num_samples = self.calculate_num_samples(self.length - len(indices))
        self.total_size = self.num_samples * self.num_replicas

    def update_indices(self, indices):
        data_size = self.length - len(indices)
        self.indices = indices
        self.num_samples = self.calculate_num_samples(data_size)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices_set = set(self.indices)
        dataset_idx = list(range(self.length))
        datalist = [item for item in dataset_idx if item not in indices_set]
        return self.shuffle_and_slice(datalist)


class D3PDistributedSamplerPrune(D3PDistributedSamplerBase):

    def __init__(self, dataset, indices, batch_size=None, padding_data=False,
                 num_replicas=None, rank=None, shuffle=True, seed=0):

        super().__init__(dataset, indices, batch_size, padding_data,
                         num_replicas, rank, shuffle, seed)

        self.num_samples = self.calculate_num_samples(len(indices))
        self.total_size = self.num_samples * self.num_replicas

    def update_indices(self, indices):
        data_size = len(indices)
        self.indices = indices
        self.num_samples = self.calculate_num_samples(data_size)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        datalist = self.indices
        return self.shuffle_and_slice(datalist)
