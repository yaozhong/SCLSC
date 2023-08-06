# Function to create data loader from anndata object
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scanpy as sc
import anndata as ad
import pandas as  pd

import torch.nn.functional as F

class GeneDataset(Dataset):

    def __init__(self, adata, k):
        genes = adata.X.toarray()
        labels = np.array(adata.obs.target)
        assert len(labels) == genes.shape[0]
        self.genes = genes
        self.labels = labels
        self.one_hot_labels = F.one_hot(torch.from_numpy(labels))
        self.n_samples = genes.shape[0]
        self.n_genes = genes.shape[1]
        # key for define label clusters
        self.n_sampling = int(len(labels)*k)
        self.representative_tensor = None
        self.representative_labels = None
        self.device = torch.device("cpu")

    def __len__(self):
        return self.n_samples


    def __getitem__(self, idx):
        gene = torch.from_numpy(self.genes[idx, :])
        label = self.one_hot_labels[idx]

        gene = gene.to(device=self.device, dtype=torch.float32)
        label = label.to(device=self.device, dtype=torch.float32)
        return gene, label


    def get_gene_profile_dim(self):
        return self.genes.shape[1]


    def to(self, device):
        self.device = device

    def create_representative_tensor(self, genes, labels, k, n_rep_mat):
        n_labels = len(np.unique(labels))
        representative_tensor = np.zeros((n_rep_mat, n_labels, genes.shape[1]))
        representative_labels = np.zeros((n_rep_mat, n_labels), np.int32)
        for i_idx in range(n_rep_mat):
            representative_mat = representative_tensor[i_idx, :, :]
            for i in np.unique(labels):
                repr_idx = np.argwhere(labels == i).flatten()
                if n_rep_mat == -1:
                    # Take mean all genes
                    repr_vector = genes[i_idx, :].mean(axis=0)
                else:
                    #Take mean of Sampling with replacement
                    repr_idx = np.random.choice(repr_idx, size=k)
                    repr_vector = genes[repr_idx, :].mean(axis=0)
                representative_mat[i, :] = repr_vector
            representative_labels[i_idx, :] = range(n_labels)
        representative_tensor = torch.from_numpy(representative_tensor)

        return representative_tensor.to(
            device=self.device, dtype=torch.float32), representative_labels



