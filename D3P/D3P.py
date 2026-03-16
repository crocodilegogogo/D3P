import numpy as np
import logging
from scipy.stats import ncx2
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
from .D3PDataset import IndexedDataset
from .D3PSampler import (
    D3PSamplerPrune,
    D3PSamplerRemain,
)


class D3P:
    def __init__(self, dataset, batchsize, args, padding_data=False,
                 shuffle=True, writer=None, logging_on=True, num_workers=4, collate_fn=None):
        # Initializes the D3P pruning class, sets up datasets, samplers, dataloaders, and loss tracking tensors.
        # logging
        self.logging_on = logging_on
        if logging_on:
            logging.info(f'Args: \n{args}')
            logging.info(f'Initializing DataPrune Class: {self.__class__.__name__}')
        self.writer = writer

        # Change the dataset to be indexed
        self.indexed_dataset = IndexedDataset(dataset)
        self.num_data = len(self.indexed_dataset)

        # Create sampler
        self.candidate = []
        self.RemainSampler = D3PSamplerRemain(
            self.indexed_dataset,
            self.candidate,
            batch_size=batchsize,
            padding_data=padding_data,
            shuffle=shuffle,
        )

        self.PruneSampler = D3PSamplerPrune(
            self.indexed_dataset,
            self.candidate,
            batch_size=batchsize,
            padding_data=padding_data,
            shuffle=shuffle,
        )

        # Create dataloader
        self.remain_dataloader = torch.utils.data.DataLoader(
            self.indexed_dataset,
            batchsize,
            sampler=self.RemainSampler,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        self.prune_dataloader = torch.utils.data.DataLoader(
            self.indexed_dataset,
            batchsize,
            sampler=self.PruneSampler,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        # Handling arguments in args
        self.s = args.s
        self.p = args.p
        self.n = args.n
        self.bin_num = args.bin_num

        # Create numpy list to store loss data
        self.loss_tensor = torch.zeros((self.n, self.num_data)).cuda()
        self.grad_scale_tensor = torch.ones(self.num_data, dtype=torch.float).cuda()
        self.data_prune = 0
        self.data_using = 0
        self.epoch = 1
        self.device = self.loss_tensor.device
        self.update = self.update_loss_all()

    def get_using_data(self):
        # Returns the count of data currently being used and the size of the pruned candidate list.
        return self.data_using, len(self.candidate)

    def get_indexed_dataset(self):
        # Returns the wrapped dataset that outputs both data and its original index.
        return self.indexed_dataset

    def get_dataloader(self):
        # Returns the dataloaders for both the remaining data and the pruned data.
        return self.remain_dataloader, self.prune_dataloader

    def data_already_prune(self):
        # Returns the total number of data samples that have been pruned.
        return self.data_prune

    def update_loss_all(self):
        # Determines whether the losses of all samples should be tracked in the current epoch based on the pruning schedule.
        start_cal = self.s - self.n + 1
        if self.epoch < start_cal:
            return False
        pc = (self.p <= self.n)
        if pc:
            return True
        epoch_relative = (self.epoch - self.s + self.n) % self.p
        update = (1 <= epoch_relative <= self.n)
        if update:
            return True
        return False

    def is_update(self):
        # Returns a boolean flag indicating if loss tracking is currently active.
        return self.update


    def update_step(self, loss_step, idx):
        # Updates the loss tensor for a batch of data during the training step and returns the scaled loss.
        self.data_using += len(idx)
        idx = idx.to(self.loss_tensor.device)
        if self.update:
            loss2_step = torch.pow(loss_step.clone().detach(), 2)
            self.loss_tensor[0, idx] = loss2_step

        loss_scale = self.grad_scale_tensor[idx]
        loss_weighted_mean = torch.sum(loss_step * loss_scale) / torch.sum(loss_scale)
        return loss_weighted_mean


    def update_forward(self, loss_step, idx):
        # Updates the loss tensor during the forward pass on pruned data.
        idx = idx.to(self.device)
        self.loss_tensor[0, idx] = loss_step.clone().detach()

    def update_epoch(self):
        # Triggers the pruning process and updates samplers at the end of an epoch if required by the schedule.
        start_to_end = (self.epoch >= self.s)
        if start_to_end and (self.epoch - self.s) % self.p == 0:
            self.grad_scale_tensor[:] = 1.
            self.prune_with_distribution()
            self.RemainSampler.update_indices(self.candidate)
            self.PruneSampler.update_indices(self.candidate)

        self.logging_epoch()
        self.clear_epoch()

    def prune_with_distribution(self):
        # Executes the core distribution-aware pruning logic to select data to discard based on the CLD metric.
        self.candidate.clear()
        loss2_tensor = torch.pow(self.loss_tensor, 2)
        somo_loss, somo_idx = torch.sort(torch.mean(loss2_tensor, dim=0, keepdim=False))

        bin_edges, peak_index, dist_fitted = self.cal_distribution(somo_loss[0].item(), somo_loss[-1].item())

        indices = torch.nonzero((somo_loss > bin_edges[0]) & (somo_loss < bin_edges[-1])).squeeze()
        left_idx, right_idx = indices[0], indices[-1] + 1

        easy_idx = somo_idx[:left_idx].tolist()
        hard_idx = somo_idx[right_idx:].tolist()
        somo_loss = somo_loss[left_idx:right_idx]
        somo_idx = somo_idx[left_idx:right_idx]

        # calculate distribution
        hist = torch.histc(somo_loss, bins=self.bin_num, min=bin_edges[0], max=bin_edges[-1])
        bin_indices = torch.bucketize(somo_loss, bin_edges[:-1], right=False) - 1
        hist_indices = {i: somo_idx[bin_indices == i] for i in range(self.bin_num)}
        data_gap = (hist - dist_fitted).int()

        # prune data
        epoch_data_prune = 0
        for i in range(len(data_gap)):
            if data_gap[i] > 0:
                ramdom_idx = torch.randperm(len(hist_indices[i]))[:data_gap[i]]
                prune_data_idx = hist_indices[i][ramdom_idx].tolist()
                self.candidate.extend(prune_data_idx)
                epoch_data_prune += len(prune_data_idx)
            elif data_gap[i] < 0 < hist[i]:
                self.grad_scale_tensor[hist_indices[i]] = dist_fitted[i] / hist[i]

        self.candidate.extend(hard_idx)
        epoch_data_prune += len(hard_idx)
        self.candidate.extend(easy_idx)
        epoch_data_prune += len(easy_idx)
        self.data_prune = epoch_data_prune

        self.logging_prune()


    def clear_epoch(self):
        # Resets the loss tracking tensors and increments the epoch counter for the next training cycle.
        self.loss_tensor = torch.roll(self.loss_tensor, shifts=1, dims=0)
        self.loss_tensor[0, :] = 0
        self.epoch += 1
        self.update = self.update_loss_all()


    def cal_distribution(self, l_cld_edge, r_cld_edge, dist_thre=0.999):
        # Calculates the theoretical chi-square distribution of the CLD metric to determine ideal data retention boundaries.
        var, mean = torch.var_mean(self.loss_tensor, dim=1, keepdim=False)
        mean_arr, var_arr = mean.cpu().numpy(), var.cpu().numpy()
        assert len(mean_arr) == len(var_arr) == self.n
        nc_arr = mean_arr ** 2 / var_arr

        pdf_upper_bound = []
        for alpha, nc in zip(var_arr, nc_arr):
            upper_bound = alpha * ncx2.ppf(dist_thre, 1, nc)
            pdf_upper_bound.append(upper_bound)

        bin_min, bin_right = min(pdf_upper_bound), max(pdf_upper_bound)
        bin_num_combined = int(bin_right / bin_min * self.bin_num)
        bin_edges_dist = np.linspace(0., bin_right, num=bin_num_combined + 1)
        bin_width_combined = bin_edges_dist[1] - bin_edges_dist[0]
        x_dist = (bin_edges_dist[:-1] + bin_edges_dist[1:]) / 2

        L_conv = bin_num_combined + (self.n - 1) * (bin_num_combined - 1)
        pad_len = 1 << ((L_conv - 1).bit_length())
        fft_pdfs = []
        for alpha, nc in zip(var_arr, nc_arr):
            pdf = ncx2.pdf(x_dist / alpha, 1, nc) / alpha
            padded_pdf = np.pad(pdf, (0, pad_len - len(pdf)))
            fft_pdfs.append(np.fft.fft(padded_pdf))

        fft_product = np.prod(fft_pdfs, axis=0)
        pdf_conved = np.fft.ifft(fft_product)[:L_conv].real
        pdf_conved *= bin_width_combined ** (self.n - 1)

        x_conv = 2 * x_dist[0] + np.arange(L_conv) * bin_width_combined
        x_scaled = x_conv / self.n
        pdf_scaled = self.n * pdf_conved

        bin_width_cld = (r_cld_edge - l_cld_edge) / self.bin_num / 8
        bin_edges_cld = np.arange(l_cld_edge, r_cld_edge, bin_width_cld)
        x_cld = (bin_edges_cld[:-1] + bin_edges_cld[1:]) / 2
        interp_func = interp1d(x_scaled, pdf_scaled, bounds_error=False, fill_value=0)
        pdf_cld = interp_func(x_cld)
        dist_num = torch.tensor(np.floor(pdf_cld * self.num_data * bin_width_cld), dtype=torch.long)
        idx_combined = torch.nonzero(dist_num >= 1).squeeze(dim=-1)
        left_bin = bin_edges_cld[idx_combined[0]]
        right_bin = bin_edges_cld[idx_combined[-1] + 1]

        bin_edges_somo = np.linspace(left_bin, right_bin, num=self.bin_num + 1)
        bin_width_somo = bin_edges_somo[1] - bin_edges_somo[0]
        x_somo = (bin_edges_somo[:-1] + bin_edges_somo[1:]) / 2
        distribution_num = torch.tensor(np.floor(interp_func(x_somo) * self.num_data * bin_width_somo),
                                        device=self.device, dtype=torch.long)
        peak_index = torch.argmax(distribution_num).item()
        bin_edges = torch.tensor(bin_edges_somo, device=self.device)

        return bin_edges, peak_index, distribution_num


    def logging_prune(self):
        # Logs the number of pruned data samples.
        if self.logging_on:
            logging.info(f'Prune {self.data_prune} data!')

    def logging_epoch(self):
        # Logs and records the data utilization percentage and statistics for the current epoch.
        data_should_use = self.num_data * self.epoch
        using_percent = self.data_using / data_should_use * 100
        if self.logging_on:
            logging.info(
                f'Should use {data_should_use} data. Actually use '
                f'{self.data_using} data, percent: {using_percent}%')

        if self.writer is not None:
            self.writer.add_scalars('data use', {'should use': data_should_use, 'already use': self.data_using},
                                    global_step=self.epoch - 1)
            self.writer.add_scalar('data using percent %', using_percent, global_step=self.epoch - 1)

