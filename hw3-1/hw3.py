# import the required libraries
import os
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

# set the random state
np.random.seed(seed=300)

# load train and test data
train = np.loadtxt(open(os.path.join('input', 'optdigits.tra'), "rb"), delimiter=",")
test =np.loadtxt(open(os.path.join('input', 'optdigits.tes'), "rb"), delimiter=",")

# Define labels for confusion matrix figure
tick_label = ['0','1','2','3','4','5','6','7','8','9']

# Slicing features and labels from train and test data
X_train = train[:,:64]
y_train = train[:,64]
X_test = test[:,:64]
y_test = test[:,64]


# spliting train set into 90% train and 10% validation set
Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.10)



# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# define a function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print(title)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)




# compute accuracy per class
def acc_per_class(conf_train, conf_test, flag=False):
	if flag:
		print('Class    train accuracy     test accuracy')
	else:
		print('Class    Before Removal     After Removal')
	for i in range(10):
		train_acc = float(conf_train[i,i])/np.sum(conf_train[i,:])
		test_acc = float(conf_test[i,i])/np.sum(conf_test[i,:])
		print(' {}       {:.4f}           {:.4f}'.format(i, train_acc, test_acc))      

#KNN classifier	 

# k values (1, 3, 5, 7, ... , 19)
k_array = np.arange(1,20,2)

# an empty list to store validation accuracies
val_scores_knn = []

# compute knn classifier accuracy for each k

best_knn = None
best_acc_knn = -1
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    val_acc_knn = knn.score(Xval, yval) # accuracy for validation set
    val_scores_knn.append(val_acc_knn)
    if val_acc_knn > best_acc_knn:
    	best_knn = knn
    	best_acc_knn = val_acc_knn

# choose the optimal k
best_k = k_array[val_scores_knn.index(max(val_scores_knn))]


print ("Best n_neighbors: {}\n".format(best_k))

# Best model accuracy on validation set
print("Validation accuracy (KNN): {:.4f}".format(best_knn.score(Xval, yval)))



# compute knn train time
print('\nComputing knn training time...')
start = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
print("n_neighbors: {}, training took {:.4f} seconds.\n".format(best_k, time.time() - start))

y_pred_train_knn = knn.predict(X_train)
conf_knn_train = confusion_matrix(y_train, y_pred_train_knn)
plot_confusion_matrix(conf_knn_train, classes=tick_label, title="Confusion Matrix of KNN (Train)")



# compute knn test time
start = time.time()
print('\nComputing knn test time...')
y_pred_test_knn = knn.predict(X_test)
print("n_neighbors: {} test took {:.4f} seconds.\n".format(best_k, time.time() - start))
print("Test accuracy (KNN): {:.4f}\n".format(knn.score(X_test, y_test)))

# plot test set confusion matrix
conf_knn_test = confusion_matrix(y_test, y_pred_test_knn)
plot_confusion_matrix(conf_knn_test, classes=tick_label, title="Confusion Matrix of KNN (Test)")





#discriminant function
# set regularization penalty
regs = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# an empty list to store validation accuracies
val_scores_sgd = []

print('Hyper-parameters tunning for linear classifier...')

best_sgd = None
best_acc_sgd = -1
for reg in regs:
    sgd = linear_model.SGDClassifier(alpha=reg)
    sgd.fit(Xtr, ytr)
    val_acc_sgd = sgd.score(Xval, yval) # accuracy for validation set
    val_scores_sgd.append(val_acc_sgd)
    if val_acc_sgd > best_acc_sgd:
    	best_sgd = sgd
    	best_acc_sgd = val_acc_sgd

# choose the best regularization penalty
best_reg = regs[val_scores_sgd.index(max(val_scores_sgd))]
print ("Best alpha: {}\n".format(best_reg))


# Validation accuracy
y_pred_val_sgd = best_sgd.predict(Xval)
print("Validation accuracy (Linear classifier): {:.4f}".format(best_sgd.score(Xval, yval)))


# compute linear classifier train time
print('\nComputing linear classifier training time...')
start = time.time()
sgd = linear_model.SGDClassifier(alpha=best_reg)
sgd.fit(X_train, y_train)
print("alpha: {}, training took {:.4f} seconds.\n".format(best_reg, time.time() - start))

# train set confusion matrix
y_pred_train_sgd = sgd.predict(X_train)
conf_sgd_train = confusion_matrix(y_train, y_pred_train_sgd)
plot_confusion_matrix(conf_sgd_train, classes=tick_label,title="Confusion Matrix of linear classifier (Train)")


# compute linear classifier test time
start = time.time()
print('\nComputing linear classifier test time...')
y_pred_test_sgd = sgd.predict(X_test)
print("alpha: {}, test took {:.4f} seconds.\n".format(best_reg, time.time() - start))
print("Test accuracy (linear classifier): {:.4f}\n".format(sgd.score(X_test, y_test)))

# plot test set confusion matrix
conf_sgd_test = confusion_matrix(y_test, y_pred_test_sgd)
plot_confusion_matrix(conf_sgd_test, classes=tick_label,title="Confusion Matrix of linear classifier (Test)")

#MLP	 


# compute mlp classifier accuracy for each set of parameters

best_mlp = None
best_acc_mlp = -1
best_hls = []
best_reg = []
hls = [64, 128, 256, (64,64), (128,128), (256,256)]
regs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for hl in hls:
    for reg in regs:
    	mlp = MLPClassifier(solver='adam', alpha=reg, hidden_layer_sizes=hl)
    	mlp.fit(Xtr, ytr)
    	val_acc_mlp = mlp.score(Xval, yval) # accuracy for validation set
    	val_scores_sgd.append(val_acc_sgd)
    	if val_acc_mlp > best_acc_mlp:
    		best_mlp = mlp
    		best_acc_mlp = val_acc_mlp
    		best_hls = hl
    		best_reg = reg

# choose the best number of components
#best_n_components = n_components_array[val_scores_lda.index(max(val_scores_lda))]

# print best hyper-parameters
print ("Best hidden_layer_sizes: {}, Best alpha: {}\n".format(best_hls, best_reg))


# Validation accuracy
print("Validation accuracy (MLP): {:.4f}".format(best_mlp.score(Xval, yval)))


# compute MLP train time
print('\nComputing MLP training time...')
start = time.time()
mlp = MLPClassifier(solver='adam', alpha=best_reg, hidden_layer_sizes=best_hls)
mlp.fit(X_train, y_train)
print("hidden_layer_sizes: {}, alpha: {}, training took {:.4f} seconds.\n".format(best_hls, best_reg,\
														 time.time() - start))


# plot train set confusion matrix
y_pred_train_mlp = mlp.predict(X_train)
conf_mlp_train = confusion_matrix(y_train, y_pred_train_mlp)
plot_confusion_matrix(conf_mlp_train, classes=tick_label,title="Confusion Matrix of MLP (Train)")


# compute MLP test time
start = time.time()
print('\nComputing MLP test time...')
y_pred_test_mlp = mlp.predict(X_test)
print("hidden_layer_sizes: {}, alpha: {}, test took {:.4f} seconds.".format(best_hls, best_reg,\
														 time.time() - start))
# print test accuracy MLP
print("Test accuracy (MLP): {:.4f}\n".format(mlp.score(X_test, y_test)))

# plot test set confusion matrix
conf_mlp_test = confusion_matrix(y_test, y_pred_test_mlp)
plot_confusion_matrix(conf_mlp_test, classes=tick_label,title="Confusion Matrix of MLP (Test)")




# compute accuracy per class KNN 

print('Accuracy per class (KNN)')
acc_per_class(conf_knn_train, conf_knn_test, flag=True)


# compute accuracy per class Linear Classifier
print('\n\n'+40*'#')
print('Accuracy per class (Linear Classifier)')
acc_per_class(conf_sgd_train, conf_sgd_test, flag=True)

# compute accuracy per class MLP
print('\n\n'+40*'#')
print('Accuracy per class (MLP)')
acc_per_class(conf_mlp_train, conf_mlp_test, flag=True)


