# 2023-05-21 single-cell task main, based on the Yusri's pipeline implementation
# 2023-07-23 revised new revision based on the clustering center

import argparse
from util import set_seed
from data_loading import *
from models import *
from visualization import *
import pickle
import time

def train(dataset_name, data_dir, save_path, gene_set, encoder_model, device, lr, margin, bs, epoch, k=100, knn_k=10):

    train_path = data_dir + "/" + dataset_name + "_train.h5ad"
    val_path   = data_dir + "/" + dataset_name + "_val.h5ad"
    test_path  = data_dir + "/" + dataset_name + "_test.h5ad"

    cell_train = sc.read_h5ad(train_path)
    cell_val = sc.read_h5ad(val_path)
    cell_test = sc.read_h5ad(test_path)

    # Metadata of highly variable gene s were stored in data_name.var["highly_variable"]
    cell_train_hvg = cell_train[:, cell_train.var["highly_variable"]]
    cell_val_hvg = cell_val[:, cell_val.var["highly_variable"]]
    cell_test_hvg = cell_test[:, cell_test.var["highly_variable"]]

    # Extract low variable genes
    cell_train_lvg = cell_train[:, ~(cell_train.var["highly_variable"])]
    cell_val_lvg = cell_val[:, ~(cell_val.var["highly_variable"])]
    cell_test_lvg = cell_test[:, ~(cell_test.var["highly_variable"])]

    if gene_set == "hvg":
          cell_train, cell_val, cell_test = cell_train_hvg, cell_val_hvg, cell_test_hvg
    elif gene_set == "lvg":
          cell_train, cell_val, cell_test = cell_train_lvg, cell_val_lvg, cell_test_lvg

    #X_train,y_train = cell_train.X.toarray(),cell_train.obs.target
    #X_val,y_val = cell_val.X.toarray(),cell_val.obs.target
    #X_test,y_test = cell_test.X.toarray(),cell_test.obs.target

    train_data = GeneDataset(cell_train, k)
    valid_data = GeneDataset(cell_val,k)
    test_data = GeneDataset(cell_test, k)
    
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=False)
    if valid_data is not None:
        valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=False)

    train_data.representative_tensor, train_data.representative_labels = train_data.create_representative_tensor(
        train_data.genes, train_data.labels, train_data.n_sampling, epoch)
    representative_tensor = train_data.representative_tensor

    print(representative_tensor)
    exit()

    train_data.to(device)
    valid_data.to(device)
    test_data.to(device)

    gene_dim = train_data.get_gene_profile_dim()
    print(f"@ used {gene_set} profile list as input, gene_dim: {gene_dim}")
   
    ##### model part ####
    if encoder_model == "MLP":
        #model = MLP(gene_dim,512,128,16)
        model = MLP(gene_dim, 512,256,128) # change with a larger MLP model
    elif encoder_model == "CNN":
        model = CNN(gene_dim, 7)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = ContrastiveLoss(margin=margin)

    early_stopper = EarlyStopper(patience=10, min_delta=0.01)
    val_step=2

    for ep in range(epoch):
        total_train_loss, total_val_loss = 0, 0
  
        if valid_data is not None:
            validation_step = (ep % val_step == 0) | (ep == (epoch - 1))
        else:
            validation_step = False
        rep_mat = representative_tensor[ep, :, :]
        rep_mat = rep_mat.to(device)

        if encoder_model == "CNN":
            rep_mat = torch.unsqueeze(rep_mat, dim=1)

        for i, (input_x, label) in enumerate(train_loader):
            model.train()
            input_x = input_x.to(device)
            label = label.to(device)

            if encoder_model == "CNN":
                input_x = torch.unsqueeze(input_x, dim=1)
                
            embed_x = model(input_x)
            # think how to change this part ? which is better ?
            embed_rep_mat = model(rep_mat)
            
            
            embed_x = embed_x[:, None, :]
            loss = criterion(embed_x, embed_rep_mat, label)

            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_loss = total_train_loss / i

        if validation_step:
            with torch.no_grad():
                for i, (input_x, label) in enumerate(valid_loader):
                    model.eval()
                    input_x = input_x.to(device)
                    label = label.to(device)
                    if encoder_model == "CNN":
                        input_x = torch.unsqueeze(input_x, dim=1)

                    embed_x = model(input_x)
                    embed_rep_mat = model(rep_mat)
                    # ADD AXIS
                    embed_x = embed_x[:, None, :]
                    loss = criterion(embed_x, embed_rep_mat, label)
                    total_val_loss += loss.item()
                total_val_loss = total_val_loss / i
                if early_stopper.early_stop(total_val_loss):
                    print(
                        "EARLY STOP||Epoch-%d, Train loss=%f, Validation loss=%f"
                        % (ep, total_train_loss, total_val_loss))
                    break

        if validation_step:
            print("Epoch-%d, Train loss=%f, Validation loss=%f" %
                  (ep, total_train_loss, total_val_loss))
        else:
            print("Epoch-%d, Train loss=%f" % (ep, total_train_loss))

    # save models
    model_info = dataset_name + "_" + encoder_model + "_lr-" + str(lr) + "_epoch-" + str(epoch) +  "_batchSize-" + str(bs) +   \
    "_margin-" + str(margin) + "_avgcell-" + str(k) + "_glist-" + gene_set
    torch.save(model.state_dict(), save_path + "/model/"+ model_info+".pt")

    X_train,y_train = cell_train.X.toarray(),cell_train.obs.target
    X_val,y_val = cell_val.X.toarray(),cell_val.obs.target
    X_test,y_test = cell_test.X.toarray(),cell_test.obs.target


    # prediction fit the training seting for the KNN first.
    print("Fitting the KNN model for the prediction ... ")
    knn = KNeighborsClassifier(n_neighbors=knn_k)

    emb_X_train = project(model, X_train, device, encoder_model)
    knn.fit(emb_X_train, y_train)
    knn_model_filename = 'KNN_'+str(knn_k)+"_" + model_info
    pickle.dump(knn, open(save_path + "/model/" + knn_model_filename + ".pkl", 'wb'))

    ######################################################################
    # visulaization results
    ######################################################################

    # doing the figure plot
    print("@ Ploting UMAP ...")
    if "batch" in dataset_name:

        group_batch_train = cell_train.obs.group_batch
        group_batch_val   = cell_val.obs.group_batch
        group_batch_test  = cell_test.obs.group_batch
        
        umap_plot_batch(encoder_model, X_test, group_batch_test, save_path, encoder_model, model, device, dataset_name+"_test" + "|" + model_info)
        umap_plot_batch(encoder_model, X_train, group_batch_train, save_path, encoder_model, model, device, dataset_name+"_train" + "|" + model_info)
        umap_plot_batch(encoder_model, X_val, group_batch_val, save_path, encoder_model, model, device, dataset_name+"_val" + "|" + model_info)

    else: 
        umap_plot_celltype(encoder_model,X_test, y_test, save_path, encoder_model, model, device, dataset_name+"_test" + "|" + model_info)
        umap_plot_celltype(encoder_model,X_train, y_train, save_path, encoder_model, model, device, dataset_name+"_train" + "|" + model_info)
        umap_plot_celltype(encoder_model,X_val, y_val, save_path, encoder_model, model, device, dataset_name+"_val" + "|" + model_info)


    ## additional prediction
    print(f"@ Calculating ACC using KNN {knn_k} ...")
    
    emb_X_test = project(model, X_test, device, encoder_model)
    y_predict_test = knn.predict(emb_X_test)
    acc = metrics.accuracy_score(y_test, y_predict_test)
    f1_score = metrics.f1_score(y_test, y_predict_test, average='macro')
    print(f"[Test] Accuracy: {acc}, F1 score: {f1_score}")


    emb_X_val = project(model, X_val, device, encoder_model)
    y_predict_val = knn.predict(emb_X_val)
    acc = metrics.accuracy_score(y_val, y_predict_val)
    f1_score = metrics.f1_score(y_val, y_predict_val, average='macro')
    print(f"[Val] Accuracy: {acc}, F1 score: {f1_score}")


if __name__ == "__main__":

        set_seed(123)
        parser = argparse.ArgumentParser(description='<Contrastive learning for the single-cell representation>')

        parser.add_argument('--encoder',       default="MLP", type=str, required=True, help='contrastive learning encoding model')
        parser.add_argument('--model_dir', action="store",   type=str, required=True,  help="directory for saving the trained model.")
        parser.add_argument('--device',       default="cuda:0", type=str, required=False, help='GPU Device(s) used for training')
        parser.add_argument('--data_dir', action="store",   type=str, required=True,  help="directory of data.")
        parser.add_argument('--output_dir', action="store",   type=str, required=True,  help="directory for saving figures.")
        parser.add_argument('--dataset_name', action="store",   type=str, required=True,  help="name of dataset.")
        parser.add_argument('--gene_set', default="all",   type=str, required=True,  help="the gene list used.[all/hvg/lvg].")

        parser.add_argument('--margin',     default=1,       type=int, required=True, help='Margins used in the contrastive training')
        parser.add_argument('--avg_sample_portion',      default=1,       type=float, required=True, help='cells used for the average representation')
        parser.add_argument('--knn_k',      default=10,       type=int, required=False, help='k of knn used model the cell type prediction')
        parser.add_argument('--lr',         default=1e-3,   type=float, required=False, help='Learning rate')
        parser.add_argument('--epoch',      default=100,       type=int, required=False, help='Training epcohs')
        parser.add_argument('--batch_size' ,default=64,      type=int,  required=False, help="batch_size of the training.")
        parser.add_argument('--dropout'    ,default=0.1,      type=float,  required=False, help="Dropout rate.")
        parser.add_argument('--workers',     default=64,       type=int, required=False, help='number of worker for data loading')


        args = parser.parse_args()

        # data loading part
        # Change this part to your dataset path

        print(" == Contrastive Learning for single-cell annotation == ")
        start_time = time.time()
        train(args.dataset_name, args.data_dir, args.output_dir, args.gene_set, args.encoder, args.device, args.lr, args.margin, args.batch_size, args.epoch, args.avg_sample_portion, args.knn_k)

        print(f" - [Running time]: {time.time()-start_time}s")


