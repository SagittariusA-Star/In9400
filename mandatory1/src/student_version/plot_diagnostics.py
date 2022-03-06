import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import h5py
import torch
from torchvision import datasets, models, transforms, utils
from RainforestDataset import get_classes_list
import PIL.Image
import os


def tail_accuracy(ts, pred, label):
    """Function computing the tail accuracy per class

    Parameters
    ----------
    ts : ndarray
        Array of acceptance probability thresholds.
    pred : ndarray
        Array of models output probability scores
    label : ndarray
        Array of ground truth labels corresponding to pred scores

    Returns
    -------
    ndarray
        Tailaccuracy
    """
    correct = pred[:, :, np.newaxis] > ts[np.newaxis, np.newaxis, :]
    correct *= correct == label[:, :, np.newaxis]

    tailac = np.nansum(correct , axis = 0).astype(np.float64)
    tailac /= np.nansum(pred[:, :, np.newaxis] > ts[np.newaxis, np.newaxis, :], axis = 0)
    return tailac


def ensure_exists(dir):
    exists = os.path.exists(dir)
    if not exists:
        os.makedirs(dir)

def evaluate_diagnostics(classwise_perf, concat_labels, concat_pred, data_root_dir, \
                        fnames, testperfs, num_epochs, figdir):
    """
    Function evaluating and plotting diagnostics.
    Parameters
    ----------
    classwise_perf : ndarray
        Classwise average precision
    concat_labels : ndarray
        Ground truth labels of validation dataset
    concat_pred : ndarray
        Prediction scores of validation dataset
    data_root_dir : str
        Data root direcory
    testperfs : ndarray
        Average precision per class and epoch
    num_epochs : int
        Number of training epochs used in traing
    figdir : str
        Figure directory to save figures in
    """
    ensure_exists(figdir)

    idx_highAP = np.argmax(classwise_perf)  # The index of the class with highest AP.
    pred   = concat_pred[:, idx_highAP]     # Prediction scores of high AP class data
    label  = concat_labels[:, idx_highAP]   # Labels of high AP class data

    idx_sorted = np.argsort(pred)[::-1]     # Index of sorted predictions

    pred   = pred[idx_sorted]   # Sort decreasing predictions 
    label  = label[idx_sorted]
    fnames = fnames[idx_sorted]

    concat_pred   = concat_pred[idx_sorted, :]   # Sort decreasing predictions 
    concat_labels = concat_labels[idx_sorted, :]

    label_names, ncls = get_classes_list()  # Get class names and number of classes
    label_names = np.array(label_names)     # 
    class_num = np.arange(ncls)

    print("------------------------------------------------")
    print(f"Highest AP class: {label_names[idx_highAP]}")
    print(f"Top-10:\n {pred[:10]}") 
    print(f"Bottom-10:\n {pred[::-1][:10]}") 
    print("------------------------------------------------")
    

    fonts = {
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
    }
    plt.rcParams.update(fonts)

    # Plot top-10 and bottom-10 images
    figt = plt.figure(figsize = (16, 4))  # Figure and axis for top-10 images
    figb = plt.figure(figsize = (16, 4))  # Figure and axis for bottom-10 images

    figt.suptitle(f"Top-10 | High AP class: {label_names[idx_highAP]}")
    figb.suptitle(f"Bottom-10 | High AP class: {label_names[idx_highAP]}")

    gst = gridspec.GridSpec(2, 5, figure = figt, wspace = 0.06, hspace = 0.01)
    gridst = [gst[i].subgridspec(1, 2, wspace = 0.0, hspace = 0.0) for i in range(10)]

    gsb = gridspec.GridSpec(2, 5, figure = figt, wspace = 0.06, hspace = 0.01)
    gridsb = [gsb[i].subgridspec(1, 2, wspace = 0.0, hspace = 0.0) for i in range(10)]


    for i in range(2):
        for j in range(5):
            idx = i * 5 + j 
            with PIL.Image.open(data_root_dir + "train-tif-v2/" + fnames[idx] + ".tif") as img:
                

                ax_dummy = figt.add_subplot(gridst[idx][:])
                plt.setp(ax_dummy.get_xticklabels(), visible = False)
                plt.setp(ax_dummy.get_yticklabels(), visible = False)
                ax_dummy.tick_params(left=False, bottom = False, labelleft = False, labelbottom = False)
                for key, spine in ax_dummy.spines.items():
                    spine.set_visible(False)

                classname = label_names[concat_labels[idx] == 1]
                
                img = np.asarray(img)# dtype = np.uint8)
                ax = figt.add_subplot(gridst[idx][0])
                ax.set_title(" " * 10 + f"Score: {pred[idx]:.8f}")
                
                
                ax.imshow(img[..., :-1])
                ax.set_xticks([])                
                ax.set_yticks([])
                ax.text(0.02, 0.05, "RGB", fontsize = 8, transform=ax.transAxes, color = "r", weight = "bold")
                
                ax = figt.add_subplot(gridst[idx][1])

                ax.imshow(img[..., -1], cmap = "gray")
                ax.set_xticks([])                
                ax.set_yticks([])
                ax.text(0.02, 0.05, r"IR", fontsize = 8, transform=ax.transAxes, color = "r", weight = "bold")
                for k, string in enumerate(classname):
                    ax.text(0.02, 0.9 - 0.07 * k, f"{string}", fontsize = 8, \
                           color = "c", transform = ax.transAxes, weight = "bold")
                    
                

            with PIL.Image.open(data_root_dir + "train-tif-v2/" + fnames[::-1][idx] + ".tif") as img:
                ax_dummy = figb.add_subplot(gridsb[idx][:])
                plt.setp(ax_dummy.get_xticklabels(), visible = False)
                plt.setp(ax_dummy.get_yticklabels(), visible = False)
                ax_dummy.tick_params(left=False, bottom = False, labelleft = False, labelbottom = False)
                for key, spine in ax_dummy.spines.items():
                    spine.set_visible(False)
        
                classname = label_names[concat_labels[::-1][idx] == 1]

                img = np.asarray(img)                
                ax = figb.add_subplot(gridsb[idx][0])
                ax.set_title(" " * 10 + f"Score: {pred[::-1][idx]:.8f}")

                ax.imshow(img[..., :-1])
                ax.set_xticks([])                
                ax.set_yticks([])
                ax.text(0.02, 0.05, "RGB", fontsize = 8, transform = ax.transAxes, color = "r", weight = "bold")
                
                ax = figb.add_subplot(gridsb[idx][1])

                ax.imshow(img[..., -1], cmap = "gray")
                ax.set_xticks([])                
                ax.set_yticks([])
                ax.text(0.02, 0.05, "IR", fontsize = 8, transform=ax.transAxes, color = "r", weight = "bold")
                for k, string in enumerate(classname):
                    ax.text(0.02, 0.9 - 0.07 * k, f"{string}", fontsize = 8, \
                            color = "c", transform = ax.transAxes, weight = "bold")
                    
    figt.savefig(figdir + "top10.pdf", bbox_inches = 'tight')
    figb.savefig(figdir + "bottom10.pdf", bbox_inches = 'tight')
    

    fonts = {
    "font.family": "serif",
    "axes.labelsize": 16,
    "font.size": 16,
    "legend.fontsize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
    }
    plt.rcParams.update(fonts)

    # Plot average precisions per class and epoch
    fig, ax = plt.subplots(figsize = (8, 5))
    colors = plt.cm.get_cmap("tab20")
    
    for i in range(ncls):
        ax.plot(np.arange(num_epochs), testperfs[:, i], color = colors.colors[i])

    ax.plot(np.arange(num_epochs), np.nanmean(testperfs, axis = 1), "r--", linewidth = 5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average precision")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(0.01, 43)
    ax.grid()
    legend = []
    ax.legend(legend + [f"AP({label_names[i]})" for i in range(ncls)] + ["Average"], ncol = 1, loc = 1)
    fig.tight_layout()
    fig.savefig(figdir + "APs.pdf", bbox_inches = 'tight')


    ts = np.linspace(0, np.max(pred), 20)   # Probalility thesholds

    tailac = tail_accuracy(ts, concat_pred, concat_labels)  # tailaccuracy

    # Plotting tailaccuracy
    fonts = {
    "font.family": "serif",
    "axes.labelsize": 20,
    "font.size": 20,
    "legend.fontsize": 12,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
    }
    plt.rcParams.update(fonts)
    fig, ax = plt.subplots(figsize = (8, 5))
    colors = plt.cm.get_cmap("tab20")
    for i in range(ncls):
        ax.plot(ts, tailac[i, :], color = colors.colors[i])
    
    ax.plot(ts, np.nanmean(tailac, axis = 0), "r--", linewidth = 5)
    
    ax.set_xlabel("Acceptance threshold t")
    ax.set_ylabel("Tail accuracy")
    ax.grid()
    ax.legend([f"{label_names[i]}" for i in range(ncls)] + ["Average"], ncol = 2, loc = "lower right")
    #ax.set_ylim(-0.1, 1.1)

    plt.savefig(figdir + "tailaccuracy.pdf", bbox_inches = 'tight')


#========================================================
#=== Update this path when you run it on your system ====
#========================================================
data_root_dir = "../../data/" 
#========================================================

###########################################################################################
#####                                     Task 1                                      #####
###########################################################################################

figdir = "../../figs/default/task1/"

# Loading data for SingleNetwork RGB
with h5py.File("diagnostics_task1.h5", "r") as infile: 
    best_epoch_1     = infile["best_epoch"][()] 
    best_measure_1   = infile["best_measure"][()] 
    trainlosses_1    = infile["trainlosses"][()] 
    testlosses_1     = infile["testlosses"][()] 
    testperfs_1      = infile["testperfs"][()] 
    concat_labels_1  = infile["concat_labels"][()]
    concat_pred_1    = infile["concat_pred"][()]
    classwise_perf_1 = infile["classwise_perf"][()]
    filenames_1      = infile["filenames"][()].astype(str)

num_epochs = testlosses_1.shape[0]

print(f"Bst epoch: {best_epoch_1 + 1}/{num_epochs}")
evaluate_diagnostics(classwise_perf_1, concat_labels_1, concat_pred_1, data_root_dir, \
                        filenames_1, testperfs_1, num_epochs, figdir)

###########################################################################################
#####                                     Task 3                                      #####
###########################################################################################
figdir = "../../figs/default/task3/"

# Loading data for TwoNetworks RGBIr
with h5py.File("diagnostics_task3.h5", "r") as infile: 
    best_epoch_3     = infile["best_epoch"][()] 
    best_measure_3   = infile["best_measure"][()] 
    trainlosses_3    = infile["trainlosses"][()] 
    testlosses_3     = infile["testlosses"][()] 
    testperfs_3      = infile["testperfs"][()] 
    concat_labels_3  = infile["concat_labels"][()]
    concat_pred_3    = infile["concat_pred"][()]
    classwise_perf_3 = infile["classwise_perf"][()]
    filenames_3      = infile["filenames"][()].astype(str)
  
num_epochs = testlosses_3.shape[0]

print(f"Bst epoch: {best_epoch_3 + 1}/{num_epochs}")
evaluate_diagnostics(classwise_perf_3, concat_labels_3, concat_pred_3, data_root_dir, \
                        filenames_3, testperfs_3, num_epochs, figdir)

###########################################################################################
#####                                     Task 4                                      #####
###########################################################################################
figdir = "../../figs/default/task4/"

# Loading data for SingleNetwork RGBIr
with h5py.File("diagnostics_task4.h5", "r") as infile: 
    best_epoch_4     = infile["best_epoch"][()] 
    best_measure_4   = infile["best_measure"][()] 
    trainlosses_4    = infile["trainlosses"][()] 
    testlosses_4     = infile["testlosses"][()] 
    testperfs_4      = infile["testperfs"][()] 
    concat_labels_4  = infile["concat_labels"][()]
    concat_pred_4    = infile["concat_pred"][()]
    classwise_perf_4 = infile["classwise_perf"][()]
    filenames_4      = infile["filenames"][()].astype(str)

num_epochs = testlosses_4.shape[0]

print(f"Bst epoch: {best_epoch_4 + 1}/{num_epochs}")
evaluate_diagnostics(classwise_perf_4, concat_labels_4, concat_pred_4, data_root_dir, \
                        filenames_4, testperfs_4, num_epochs, figdir)



figdir = "../../figs/default/"
ensure_exists(figdir)

fonts = {
"font.family": "serif",
"axes.labelsize": 16,
"font.size": 16,
"legend.fontsize": 16,
"xtick.labelsize": 16,
"ytick.labelsize": 16
}
plt.rcParams.update(fonts)


label_names, ncls = get_classes_list()
label_names = np.array(label_names)
class_num = np.arange(ncls)

# Plotting Average Precision per class

fig, ax = plt.subplots(figsize = (16, 5))

ax.axhline(np.mean(classwise_perf_1), color = "b", linestyle = "dashed")
ax.axhline(np.mean(classwise_perf_3), color = "orange", linestyle = "dashed")
ax.axhline(np.mean(classwise_perf_4), color = "g", linestyle = "dashed")
ax.bar(class_num, classwise_perf_1, label = "SingleNetwork RGB", width = 0.9)
ax.bar(class_num, classwise_perf_3, label = "TwoNetworks RGBIr", width = 0.7)
ax.bar(class_num, classwise_perf_4, label = "SingleNetwork RGBIr", width = 0.5)
#ax.scatter(idx_highAP, np.max(classwise_perf), label = "Max AP", c = "r")
ax.set_xticks(class_num)
ax.set_xticklabels(label_names, rotation = 30)
ax.set_xlabel("Class labels")
ax.set_ylabel("Average precision")
ax.legend(loc = "upper center")
fig.tight_layout()
plt.savefig(figdir + "best_classwise_perf.pdf", bbox_inches = "tight")


# Plot training and validation loss
fig, ax = plt.subplots(figsize = (8, 5))
ax.plot(np.arange(num_epochs), trainlosses_1, "r", label = "Train loss SingleNetwork RGB")
ax.plot(np.arange(num_epochs), testlosses_1, "r--",  label = "Test loss SingleNetwork RGB")

ax.plot(np.arange(num_epochs), trainlosses_3, "b", label = "Train loss TwoNetworks RGBIr")
ax.plot(np.arange(num_epochs), testlosses_3, "b--",  label = "Test loss TwoNetworks RGBIr")

ax.plot(np.arange(num_epochs), trainlosses_4, "g", label = "Train loss SingleNetwork RGBIr")
ax.plot(np.arange(num_epochs), testlosses_4, "g--",  label = "Test loss SingleNetwork RGBIr")

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid()
ax.legend()
fig.tight_layout()
plt.savefig(figdir + "loss.pdf", bbox_inches = 'tight')
