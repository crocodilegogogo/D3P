from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ori_data = self.dataset[idx]
        return ori_data, idx
