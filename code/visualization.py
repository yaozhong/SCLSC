# visualization
import matplotlib.pyplot as plt
import umap
import pandas as  pd

from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from models import project

#def accuarcy_knn(k=10):


def umap_plot_batch(encoder_model, X, group_batch, save_path, model_info, encoder, device, data_profile):

    f = plt.figure(figsize=(16,8), dpi=300)
    ax_1 = f.add_subplot(121)
    ax_2 = f.add_subplot(122)

    ax_1.set_title('(batch) UMAP of raw data')
    ax_2.set_title('(batch) UMAP of embedded data (CL+MLP)')

    # The UMAP representation of training data embedding
    umap_reducer = umap.UMAP()
    X_train_umap = umap_reducer.fit_transform(X)
    ax_1.scatter(X_train_umap[:, 0],X_train_umap[:, 1],c=group_batch,alpha=0.8,s=0.5)

    # The UMAP representation of embedded training data embedding
    emb_X_train = project(encoder, X, device, encoder_model)
    umap_reducer = umap.UMAP()
    emb_X_train_umap = umap_reducer.fit_transform(emb_X_train)
    ax_2.scatter(emb_X_train_umap[:, 0],emb_X_train_umap[:, 1], c=group_batch, alpha=0.8,s=0.5)

    f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")



def umap_plot_celltype(encoder_model, X, y, save_path, model_info, encoder, device, data_profile):

	f = plt.figure(figsize=(16,8), dpi=300)
	ax_1 = f.add_subplot(121)
	ax_2 = f.add_subplot(122)

	ax_1.set_title('UMAP  of raw data')
	ax_2.set_title('UMAP  of embedded data (CL+MLP)')

	# The UMAP representation of training data embedding
	umap_reducer = umap.UMAP()
	X_train_umap = umap_reducer.fit_transform(X)
	ax_1.scatter(X_train_umap[:, 0],X_train_umap[:, 1],c=y,alpha=0.8,s=0.5)

	# The UMAP representation of embedded training data embedding
	emb_X_train = project(encoder, X, device, encoder_model)
	umap_reducer = umap.UMAP()
	emb_X_train_umap = umap_reducer.fit_transform(emb_X_train)
	ax_2.scatter(emb_X_train_umap[:, 0], emb_X_train_umap[:, 1], c=y, alpha=0.8,s=0.5)

	f.savefig(save_path + "/figures/" + model_info + "/umap_single_cell_" + data_profile +".png")

	# Predict  the annotation from low dimensional embedding space
	"""
	knn = KNeighborsClassifier(n_neighbors=10)
	knn.fit(emb_X_train, y_train)
	y_predict = knn.predict(emb_X_test)

	acc = metrics.accuracy_score(y_test, y_predict)
	f1_score = metrics.f1_score(y_test, y_predict, average='macro')
	print("Accuracy: %f, F1 score: %f" % (acc, f1_score))
	"""

	