import argparse
import numpy as np
import os
import glob
from dataset.image_dataset import ImgDataset
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.ResNet import ResNet50_encoder
import torchvision.transforms as T



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        required=True,
        nargs='*',
        type=str,
        default=None,
        help="Data directory containing the second pair of image dataset",
    )
    parser.add_argument(
        "--labels",
        required=False,
        nargs='*',
        type=str,
        default=None,
        help="labels for legend seperated by spcae (e.g. Real_Data Generated_Data ...)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="number of plotted samples for each class",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Batch Size, default = 50",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="ResNet50", 
        choices=['ResNet50', 'VGG16'],
        help="Choose the backbone encoder.",
    )
    parser.add_argument(
        "--visualize_imgs",
        action="store_true",
        help="Visualize the images, not only scatter plot",
    )
    parser.add_argument(
        "--fid",
        action="store_true",
        help="Compute FID",
    )
    args = parser.parse_args()
    return args



def get_data(img_list, backbone ='ResNet50', batch_size = 50 ):
    transform = None
    if backbone == "ResNet50":
        transform =  torch.nn.Sequential(
            T.Resize(232, antialias=True),
            T.Normalize([0.485, 0.456, 0.406], [0.299, 0.224, 0.225]),)
        

    dataset = ImgDataset(img_list, transform=transform)
    data_loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)
    return data_loader    

def get_model(backbone = 'ResNet50'):
    if backbone == 'ResNet50':
        model = ResNet50_encoder()
        model = model.cuda()
        model.eval()
        return model
    else:
        print("No model has been loaded.")
        return None

def get_embeddings(data_loader, model, run_identifier = "Real"):
    features = None
    for batch in tqdm(data_loader, desc=f"Running the model on {run_identifier}"):
        images = batch.cuda()
        output = model(images).detach().cpu().numpy()

        if features is None:
            features = output
        else:
            features = np.concatenate([features, output])
    return features

def get_cmap():
    return np.array(([1.0, 0.0, 0.0], 
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]))

def scatter_plot(features, labels, names = None):
    # initialize a matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap = get_cmap()
    
    if names == None:
        names = [f'dataset {i + 1}' for i in range(len(np.unique(labels))) ]


    # for every class, we'll add a scatter plot separately
    for label in np.unique(labels):
        # find the samples of the current class in the data
        indices = features[labels == label]

        # extract the coordinates of the points of this class only
        current_tx = indices[:, 0]
        current_ty = indices[:, 1]

        # add a scatter plot with the corresponding color and label
        text =names[int(label)]
        
        ax.scatter(current_tx, current_ty, c=cmap[int(label)].reshape(1,-1), label=text, alpha=0.5)

    # build a legend using the labels we set previously
    ax.legend(loc="best")

    # finally, show the plot
    plt.show()

def normalize_features(x):
    value_range = np.max(x) - np.min(x)
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def visualize(args):

    datasets = args.datasets
    datasets_names = args.labels

    if datasets_names:
        assert len(datasets) == len(datasets_names)
    n_samples = args.n_samples
    backbone = args.backbone
    batch_size = args.batch_size
    model = get_model(backbone)


    combined_features, combined_labels, combined_img_paths = [], [], []
    for i, dataset in enumerate(datasets):

        imgs_paths = glob.glob(os.path.join(os.path.abspath(dataset),
                                            "*"))
        imgs_paths = np.random.permutation(imgs_paths).tolist()
        if n_samples:
            imgs_paths = imgs_paths[:n_samples]
        ImgDataset_loader = get_data(imgs_paths,backbone, batch_size = batch_size)
        imgs_embeddings = get_embeddings(ImgDataset_loader, model, run_identifier = "Real")
        combined_features.append(imgs_embeddings)
        combined_labels.append(np.ones(imgs_embeddings.shape[0]) * i)
        combined_img_paths.extend(imgs_paths)

    combined_features = np.concatenate(combined_features)
    combined_labels = np.concatenate(combined_labels)
    
    perplexity = combined_features.shape[0] // len(datasets)
    features = TSNE(n_components=2, perplexity=perplexity).fit_transform(
        combined_features
    )
    features[:, 0] = normalize_features(features[:, 0])
    features[:, 1] = normalize_features(features[:, 1])
    scatter_plot(features, combined_labels,datasets_names )



if __name__ == "__main__":
    args = get_args()
    visualize(args)