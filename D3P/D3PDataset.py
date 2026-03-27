from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        # Initializes the dataset wrapper to store the original dataset.
        self.dataset = dataset

    def __len__(self):
        # Returns the total number of samples in the wrapped dataset.
        return len(self.dataset)

    def __getitem__(self, idx):
        # Returns the original data sample along with its index.
        ori_data = self.dataset[idx]
        return ori_data, idx
