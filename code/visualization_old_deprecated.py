# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as  pd
import numpy as np
import os

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
import argparse
from util import set_seed
from data_loading import *
from models import *


def util_plot(X, y, ax,cmaps):

    for i in np.unique(y):
        ax.scatter(X[y == i, 0],
                   X[y == i, 1],
                   label=i,
                   color=cmaps[i],
                   alpha=0.8,s=0.5)


def umap_plot_batch2(encoder_model, X_train, X_test, y_train, y_test,
                    save_path, model_info, encoder, device, data_profile):

    f, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16),sharey=True)

    ax_1=axes[0,0]
    ax_2=axes[0,1]
    ax_3=axes[1,0]
    ax_4=axes[1,1]

    ax_1.set_title('(batch) UMAP  of raw training data')
    ax_2.set_title('(batch) UMAP  of raw test data')

    ax_3.set_title('(batch) UMAP  of embedded training data (CL+MLP)')
    ax_4.set_title('(batch) UMAP  of embedded test data (CL+MLP)')



    # The UMAP representation of training data embedding
    umap_reducer = umap.UMAP().fit(np.vstack([X_train,X_test]))

    X_train_umap = umap_reducer.transform(X_train)
    unique_target = np.unique(np.concatenate((y_train,y_test)))
    cmaps=sns.color_palette("husl",len(unique_target))
    util_plot(X_train_umap,y_train,ax_1,cmaps)

    X_test_umap = umap_reducer.transform(X_test)
    util_plot(X_test_umap,y_test,ax_2,cmaps)

    # The UMAP representation of embedded training data embedding
    emb_X_train = project(encoder, X_train, device, encoder_model)
    emb_X_test = project(encoder, X_test, device, encoder_model)

    umap_reducer = umap.UMAP().fit(np.vstack([emb_X_train,emb_X_test]))
    emb_X_train_umap = umap_reducer.transform(emb_X_train)
    util_plot(emb_X_train_umap,y_train,ax_3,cmaps)

    emb_X_test_umap = umap_reducer.transform(emb_X_test)
    util_plot(emb_X_test_umap,y_test,ax_4,cmaps)

    plt.tight_layout()
    f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")
    return umap_reducer



def umap_plot_celltype2(encoder_model, X_train, X_test, y_train, y_test,
                      save_path, model_info, encoder, device, data_profile):

  f, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 16),sharey=True)

  ax_1=axes[0,0]
  ax_2=axes[0,1]
  ax_3=axes[1,0]
  ax_4=axes[1,1]

  ax_1.set_title('UMAP  of raw training data')
  ax_2.set_title('UMAP  of raw test data')

  ax_3.set_title('UMAP  of embedded training data (CL+MLP)')
  ax_4.set_title('UMAP  of embedded test data (CL+MLP)')



  # The UMAP representation of training data embedding
  umap_reducer = umap.UMAP().fit(np.vstack([X_train,X_test]))

  X_train_umap = umap_reducer.transform(X_train)
  unique_target = np.unique(np.concatenate((y_train,y_test)))
  cmaps=sns.color_palette("husl",len(unique_target))
  util_plot(X_train_umap,y_train,ax_1,cmaps)

  X_test_umap = umap_reducer.transform(X_test)
  util_plot(X_test_umap,y_test,ax_2,cmaps)

  # The UMAP representation of embedded training data embedding
  emb_X_train = project(encoder, X_train, device, encoder_model)
  emb_X_test = project(encoder, X_test, device, encoder_model)

  umap_reducer = umap.UMAP().fit(np.vstack([emb_X_train,emb_X_test]))
  emb_X_train_umap = umap_reducer.transform(emb_X_train)
  util_plot(emb_X_train_umap,y_train,ax_3,cmaps)

  emb_X_test_umap = umap_reducer.transform(emb_X_test)
  util_plot(emb_X_test_umap,y_test,ax_4,cmaps)

  plt.tight_layout()
  f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")


if __name__ == "__main__":

        set_seed(100)
        parser = argparse.ArgumentParser(description='<Visualization Contrastive learning for the single-cell representation>')

        parser.add_argument('--encoder_path', type=str, required=True, help='contrastive learning encoding model')
        parser.add_argument('--encoder',       default="MLP", type=str, required=True, help='contrastive learning encoding model')

        parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for loading the trained model.")
        parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
        parser.add_argument('--data_dir', action="store",   type=str, required=True,  help="directory of data.")
        parser.add_argument('--output_dir', action="store",   type=str, required=True,  help="directory for saving figures.")
        parser.add_argument('--dataset_name', action="store",   type=str, required=True,  help="name of dataset.")
        parser.add_argument('--gene_set', default="all",   type=str, required=True,  help="the gene list used.[all/hvg/lvg].")
        parser.add_argument('--workers',     default=64,       type=int, required=False, help='number of worker for data loading')


        args = parser.parse_args()

        # data loading part
        # Change this part to your dataset path

        print(" == Visualization Contrastive Learning for single-cell annotation == ")
        data_dir = args.data_dir
        dataset_name = args.dataset_name
        encoder_model = args.encoder
        device = args.device
        train_path = data_dir + "/" + dataset_name + "_train.h5ad"
        test_path  = data_dir + "/" + dataset_name + "_test.h5ad"

        cell_train = sc.read_h5ad(train_path)
        cell_test = sc.read_h5ad(test_path)

        if args.gene_set == "hvg":
          # cell_train, cell_val, cell_test = cell_train_hvg, cell_val_hvg, cell_test_hvg
          cell_train = cell_train[:, cell_train.var["highly_variable"]]
          cell_test = cell_test[:, cell_test.var["highly_variable"]]
        elif args.gene_set == "lvg":
          # cell_train, cell_val, cell_test = cell_train_lvg, cell_val_lvg, cell_test_lvg
          cell_train = cell_train[:, ~(cell_train.var["highly_variable"])]
          cell_test = cell_test[:, ~(cell_test.var["highly_variable"])]

        X_train, y_train = cell_train.X.toarray(), cell_train.obs.target
        X_test, y_test = cell_test.X.toarray(), cell_test.obs.target
        gene_dim = X_train.shape[1]
        if encoder_model == "MLP":
          model = MLP(gene_dim,512,128,16)
          #model = MLP(gene_dim,128,64,16)
        elif encoder_model == "CNN":
          model = CNN(gene_dim, 7)

        model.to(device)
        model.load_state_dict(torch.load(args.encoder_path))
        model_info = os.path.splitext(os.path.basename(args.encoder_path))[0]
        print("@ Ploting UMAP ...")
        if "batch" in dataset_name:

            group_batch_train = cell_train.obs.group_batch
            group_batch_test  = cell_test.obs.group_batch


            umap_plot_batch2(encoder_model, X_train, X_test, y_train, y_test,
                          args.output_dir, encoder_model, model, device, dataset_name+"_train+test" + "|" + model_info)
        else:
            umap_plot_celltype2(encoder_model,X_train, X_test, y_train, y_test,
                          args.output_dir, encoder_model, model, device, dataset_name+"_train+test" + "|" + model_info)

