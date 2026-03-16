import math
import numpy as np
import logging
from scipy.stats import ncx2
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from .D3PDataset import IndexedDataset
from .D3PSampler import (
    D3PDistributedSamplerPrune,
    D3PDistributedSamplerRemain
)


class D3PDistributed:
    def __init__(self, dataset, batchsize, args, padding_data=False, shuffle=True,
                 writer=None, logging_on=True, num_workers=4, num_replicas=None, rank=None, collate_fn=None):
        # Initializes the distributed D3P pruning class with multi-GPU synchronization support.
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
        if args.distributed:
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()
        else:
            num_replicas = 1
            rank = 0

        self.num_replicas = num_replicas
        self.rank = rank
        self.device = torch.device(f'cuda:{args.gpu}') if args.distributed else torch.device('cpu')

        self.num_data_train = int(math.ceil(self.num_data / self.num_replicas)) * self.num_replicas

        # Create dataloader
        self.RemainSampler = D3PDistributedSamplerRemain(
            dataset=self.indexed_dataset,
            indices=self.candidate,
            batch_size=batchsize,
            padding_data=padding_data,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=args.seed
        )

        self.PruneSampler = D3PDistributedSamplerPrune(
            dataset=self.indexed_dataset,
            indices=self.candidate,
            batch_size=batchsize,
            padding_data=padding_data,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=args.seed
        )

        self.remain_dataloader = torch.utils.data.DataLoader(
            self.indexed_dataset,
            batch_size=batchsize,
            sampler=self.RemainSampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        self.prune_dataloader = torch.utils.data.DataLoader(
            self.indexed_dataset,
            batch_size=batchsize,
            sampler=self.PruneSampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Handling arguments in args
        self.s = args.s
        self.p = args.p
        self.n = args.n
        self.bin_num = args.bin_num

        # Create tensor to store loss data
        if self.rank == 0:
            self.loss_cache = torch.zeros((self.n, self.num_data)).to(self.device)
        else:
            self.loss_cache = None
        self.loss_tensor = torch.zeros(self.num_data).to(self.device)
        self.loss2_counter = torch.zeros(self.num_data).to(self.device)
        self.grad_scale_tensor = torch.ones(self.num_data, dtype=torch.float).to(self.device)
        self.data_using = 0
        self.epoch = 1
        self.update = self.update_loss_all()

    def sync_candidate(self):
        # Synchronizes the list of pruned data candidates across all distributed processes.
        candidate_tensor = torch.tensor(self.candidate, dtype=torch.long).to(self.device)
        candidate_length = torch.tensor([len(self.candidate)], dtype=torch.long).to(self.device)
        dist.broadcast(candidate_length, src=0)
        if len(self.candidate) < candidate_length.item():
            padding = torch.zeros(candidate_length.item() - len(self.candidate), dtype=torch.long).to(self.device)
            candidate_tensor = torch.cat([candidate_tensor, padding], dim=0)
        elif len(self.candidate) > candidate_length.item():
            candidate_tensor = candidate_tensor[:candidate_length.item()]
        dist.broadcast(candidate_tensor, src=0)
        if self.rank != 0:
            self.candidate = candidate_tensor.cpu().tolist()

    def sync_grad_scale(self):
        # Synchronizes the gradient scaling tensor across all distributed processes.
        dist.broadcast(self.grad_scale_tensor, src=0)

    def gather_loss2_array(self):
        # Aggregates the squared loss arrays from all distributed processes.
        dist.all_reduce(self.loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.loss2_counter, op=dist.ReduceOp.SUM)

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
        # Determines whether the losses should be tracked and updated in the current distributed epoch.
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
        # Updates the loss tensor with gradient scaling for a batch during the distributed training step.
        self.data_using += len(idx)
        idx = idx.to(self.device)
        if self.update:
            self.loss_tensor[idx] += loss_step.clone().detach()
            self.loss2_counter[idx] += 1

        loss_scale = self.grad_scale_tensor[idx]
        loss_weighted_mean = torch.sum(loss_step * loss_scale) / torch.sum(loss_scale)
        return loss_weighted_mean

    def update_forward(self, loss_step, idx):
        # Updates the loss tensor during the forward pass on pruned data in the distributed setup.
        idx = idx.to(self.device)
        self.loss_tensor[idx] += loss_step.clone().detach()
        self.loss2_counter[idx] += 1

    def update_epoch(self):
        # Aggregates losses across GPUs, performs distribution-aware pruning, and synchronizes samplers at the epoch's end.
        start_to_end = (self.epoch >= self.s)
        update_this_epoch = start_to_end and ((self.epoch - self.s) % self.p == 0)
        self.gather_loss2_array()
        if self.rank == 0:
            self.loss_cache[0, :] = self.loss_tensor / self.loss2_counter
            if update_this_epoch:
                self.grad_scale_tensor[:] = 1.
                self.prune_with_distribution()
            self.logging_epoch_all()

        if update_this_epoch:
            self.sync_candidate()
            self.sync_grad_scale()
            self.RemainSampler.update_indices(self.candidate)
            self.PruneSampler.update_indices(self.candidate)

        self.clear_epoch()

    def prune_with_distribution(self):
        # Executes the core distribution-aware pruning logic using the globally aggregated loss.
        self.candidate.clear()
        loss2_tensor = torch.pow(self.loss_cache, 2)
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
        # Clears loss caches and increments the epoch counter across all distributed processes.
        if self.rank == 0:
            self.loss_cache = torch.roll(self.loss_cache, shifts=1, dims=0)
            self.loss_cache[0, :] = 0
        self.loss_tensor[:] = 0
        self.loss2_counter[:] = 0
        self.epoch += 1
        self.update = self.update_loss_all()


    def cal_distribution(self, l_cld_edge, r_cld_edge, dist_thre=0.999):
        # Calculates the ideal chi-square distribution of the CLD metric for the distributed data.
        var, mean = torch.var_mean(self.loss_cache, dim=1, keepdim=False)
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

    def logging_epoch_all(self):
        # Logs and records the global data utilization percentage and statistics for the distributed epoch.
        data_should_use = self.num_data_train * self.epoch
        using_percent = self.data_using * self.num_replicas / data_should_use * 100
        if self.logging_on:
            logging.info(
                f'Should use {data_should_use} data. Actually use '
                f'{self.data_using * self.num_replicas} data, percent: {using_percent}%')

        if self.writer is not None:
            self.writer.add_scalars('data_use',
                                    {'should use': data_should_use, 'already use': self.data_using},
                                    global_step=self.epoch - 1)
            self.writer.add_scalar('data_using_percent_%', using_percent, global_step=self.epoch - 1)
