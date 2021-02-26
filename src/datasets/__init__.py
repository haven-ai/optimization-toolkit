import torchvision
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from sklearn import metrics
from src import utils as ut
from torch.utils.data import Dataset
import tqdm

import torchvision
import torch
from haven import haven_utils as hu
import numpy as np
from torchvision.transforms import transforms
from sklearn.utils import shuffle
from PIL import Image
from . import cityscapes
from . import pascal

from src import utils as ut
import os
import os
import numpy as np

import random
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image


def get_semseg_dataset(name, split, datadir, exp_dict, dataset_size=None):

    if name == 'pascal':
        # datadir = os.path.join('/mnt/datasets/public/issam/')
        
        dataset = pascal.Pascal(datadir, split=split, supervision='full', 
                                exp_dict=exp_dict,
                                sbd=False)

        if dataset_size is not None and dataset_size.get(split, 'all') != 'all':
            dataset.dataset.images = dataset.dataset.images[:dataset_size[split]]

    elif name == "cityscapes":
        dataset = cityscapes.CityScapes(split=split, exp_dict=exp_dict, datadir=datadir)
        if dataset_size is not None and dataset_size.get(split, 'all') != 'all':
            dataset.dataset.images = dataset.dataset.images[:dataset_size[split]]
     
    else:
        raise ValueError('dataset %s not found' % name)

    print(split, ':', len(dataset))
    return dataset

def get_random(y_list, x_list):
    with hu.random_seed(1):
        yi = np.random.choice(y_list)
        x_tmp = x_list[y_list == yi]
        xi = np.random.choice(x_tmp)

    return yi, xi

def get_median(y_list, x_list):
    tmp = y_list
    mid = max(0, len(tmp)//2 - 1)
    yi = tmp[mid]
    tmp = x_list[y_list == yi]
    mid = max(0, len(tmp)//2 - 1)
    xi = tmp[mid]

    return yi, xi

class BatchDictator:
    def __init__(self, split, n_classes):
        self.n_classes = n_classes 

        # irn stuff
        self.scales = (1.0,)
        if split == 'train':
            self.resize_long = (320, 640)
            self.rescale = None
            self.crop_size = 512
            self.img_normal = TorchvisionNormalize()
            self.hor_flip = True
            self.crop_method = "random"
            self.to_torch = True

        elif split == 'val':
            self.resize_long = None
            self.rescale = None
            self.crop_size = 512
            self.img_normal = TorchvisionNormalize()
            self.hor_flip = None
            self.crop_method = None
            self.to_torch = True

    def get_batch(self, img_pil, mask_pil, transforms, idx, class_pil=None, 
                       void=255, inst_pil=None, name=None, points=None):

        if inst_pil is not None:
            mask = np.array(inst_pil)
            mask_void = mask == void
            mask[mask_void] = 0
        else:
            mask = np.array(mask_pil)
            mask_void = mask == void
            mask[mask_void] = 0
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        # if num_objs == 0:
        #     raise ValueError('should have car')

        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        if inst_pil is not None:
            mask_color = np.array(mask_pil)
            labels = []
            for m in masks:
                labels += [(mask_color*m).max()]
            labels = torch.LongTensor(labels)
        else:
            labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        if num_objs == 0:
            area = torch.tensor([])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if transforms is not None and len(boxes):
            img, target = transforms(img_pil, target)
        else:
            import torchvision
            img = torchvision.transforms.ToTensor()(img_pil)
            target = None

        labels_tensor = torch.zeros(self.n_classes-1)
        labels_tensor[labels-1] = 1
        # return img, target
        batch_dict = {'images':img, 
                    'mask_pil':mask_pil,
                    'image_pil':img_pil,
                    'original':np.array(img_pil),
                    'inst_pil':inst_pil,
                    'label':labels_tensor,
                    'points':points,
                    'mask_void':torch.FloatTensor(mask_void),
                    'targets': target,
                    'meta':{'index':idx, 'name':name}}

        return batch_dict
        
    def transform_img(self, img_org):
        img = img_org

        if self.resize_long:
            img = imutils.random_resize_long(img, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            img = imutils.random_scale(img, scale_range=self.rescale, order=3)

        if self.img_normal:
            img = self.img_normal(img)

        if self.hor_flip:
            img = imutils.random_lr_flip(img)

        if self.crop_size:
            if self.crop_method == "random":
                img = imutils.random_crop(img, self.crop_size, 0)
            else:
                img = imutils.top_left_crop(img, self.crop_size, 0)

        if self.to_torch:
            img = imutils.HWC_to_CHW(img)

        return img

    def get_cam_batch(self, img_org):
        batch = {}
        img = self.transform_img(img_org)
        ms_img_list = self.get_multi_scale(img_org)
        
        batch['img'] =  torch.from_numpy(img)
        batch['img_msf'] =  ms_img_list
        batch['size'] = (img_org.shape[0], img_org.shape[1])

        return batch

    def get_multi_scale(self, img_org):
        img = img_org

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                s_img = imutils.pil_rescale(img, s, order=3)

            s_img = self.img_normal(s_img)
            s_img = imutils.HWC_to_CHW(s_img)
            ms_img_list.append(np.stack([s_img, np.flip(s_img, -1)], axis=0))

        if len(self.scales) == 1:
            ms_img_list = ms_img_list[0]

        out =  torch.from_numpy(ms_img_list)

        return out



def get_dataset(dataset_name, split, datadir, exp_dict):
    train_flag = True if split == 'train' else False
    if dataset_name in ['pascal']:
        return get_semseg_dataset(dataset_name, split, datadir, exp_dict, dataset_size=None)

    if dataset_name in ['B', 'C']:
        bias = 1; 
        scaling = 10; 
        sparsity = 10; 
        solutionSparsity = 0.1;
        n = 1000

        if dataset_name == 'C':
            p = 100
        if dataset_name == 'B':
            p = 10000
            
        A = np.random.randn(n,p)+bias;
        A = A.dot(np.diag(scaling* np.random.randn(p)))
        A = A * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        b = np.sign(A.dot(w));
        b = b * np.sign(np.random.rand(n)-0.1);
        labels = np.unique(b)
        A = A / np.linalg.norm(A, axis=1)[:, None].clip(min=1e-6)
        A = A * 2
        b[b==labels[0]] = 0
        b[b==labels[1]] = 1

        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))

        return DatasetWrapper(dataset, split=split)

    if dataset_name == 'tiny_imagenet':
        if train_flag:
            transform_train = transforms.Compose([
                # transforms.Resize(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            # define dataloader
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                        transform=transform_train)

        else:
            transform_test = transforms.Compose([
                # transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                       transform=transform_test)

    if dataset_name == 'imagenette2-160':
        if train_flag:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            # define dataloader
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                        transform=transform_train)

        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                       transform=transform_test)

    if dataset_name == 'imagewoof2-160':
        if train_flag:
            transform_train = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
            # define dataloader
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                        transform=transform_train)

        else:
            transform_test = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            dataset = torchvision.datasets.ImageFolder(root=datadir, 
                                                       transform=transform_test)

    if dataset_name == "mnist":
        view = torchvision.transforms.Lambda(lambda x: x.view(-1).view(784))
        dataset = torchvision.datasets.MNIST(datadir, train=train_flag,
                               download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.5,), (0.5,)),
                                   view
                               ])
                               )

    if dataset_name == "cifar10":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name == "cifar100":
        transform_function = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        dataset = torchvision.datasets.CIFAR100(
            root=datadir,
            train=train_flag,
            download=True,
            transform=transform_function)

    if dataset_name in ['B', 'C']:
        bias = 1; 
        scaling = 10; 
        sparsity = 10; 
        solutionSparsity = 0.1;
        n = 1000

        if dataset_name == 'C':
            p = 100
        if dataset_name == 'B':
            p = 10000
            
        A = np.random.randn(n,p)+bias;
        A = A.dot(np.diag(scaling* np.random.randn(p)))
        A = A * (np.random.rand(n,p) < (sparsity*np.log(n)/n));
        w = np.random.randn(p) * (np.random.rand(p) < solutionSparsity);

        b = np.sign(A.dot(w));
        b = b * np.sign(np.random.rand(n)-0.1);
        labels = np.unique(b)
        A = A / np.linalg.norm(A, axis=1)[:, None].clip(min=1e-6)
        A = A * 2
        b[b==labels[0]] = 0
        b[b==labels[1]] = 1
        # squared_max, squared_min = compute_max_eta_squared_loss(A)
        # logistic_max, logistic_min = compute_max_eta_logistic_loss(A)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(A), torch.FloatTensor(b))

        return DatasetWrapper(dataset, split=split)

        
    if dataset_name in ["mushrooms", "w8a",
                        "rcv1", "ijcnn", 'a1a','a2a',
                        "mushrooms_convex", "w8a_convex",
                        "rcv1_convex", "ijcnn_convex", 'a1a_convex'
                        , 'a2a_convex']:

        sigma_dict = {"mushrooms": 0.5,
                      "w8a":20.0,
                      "rcv1":0.25 ,
                      "ijcnn":0.05}

        X, y = load_libsvm(dataset_name.replace('_convex', ''), 
                           data_dir=datadir)

        labels = np.unique(y)

        y[y==labels[0]] = 0
        y[y==labels[1]] = 1
        # splits used in experiments
        splits = train_test_split(X, y, test_size=0.2, shuffle=True, 
                    random_state=9513451)
        X_train, X_test, Y_train, Y_test = splits

        if "_convex" in dataset_name:
            if train_flag:
                # training set
                X_train = torch.FloatTensor(X_train.toarray())
                Y_train = torch.FloatTensor(Y_train)
                dataset = torch.utils.data.TensorDataset(X_train, Y_train)
            else:
                # test set
                X_test = torch.FloatTensor(X_test.toarray())
                Y_test = torch.FloatTensor(Y_test)
                dataset = torch.utils.data.TensorDataset(X_test, Y_test)

            return DatasetWrapper(dataset, split=split)

        if train_flag:
            # fname_rbf = "%s/rbf_%s_%s_train.pkl" % (datadir, dataset_name, sigma_dict[dataset_name])
            fname_rbf = "%s/rbf_%s_%s_train.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_train_X = np.load(fname_rbf)
            else:
                k_train_X = rbf_kernel(X_train, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_train_X)
                print('%s saved' % fname_rbf)

            X_train = k_train_X
            X_train = torch.FloatTensor(X_train)
            Y_train = torch.LongTensor(Y_train)

            dataset = torch.utils.data.TensorDataset(X_train, Y_train)

        else:
            fname_rbf = "%s/rbf_%s_%s_test.npy" % (datadir, dataset_name, sigma_dict[dataset_name])
            if os.path.exists(fname_rbf):
                k_test_X = np.load(fname_rbf)
            else:
                k_test_X = rbf_kernel(X_test, X_train, sigma_dict[dataset_name])
                np.save(fname_rbf, k_test_X)
                print('%s saved' % fname_rbf)

            X_test = k_test_X
            X_test = torch.FloatTensor(X_test)
            Y_test = torch.LongTensor(Y_test)

            dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    if dataset_name == "synthetic":
        margin = exp_dict["margin"]

        X, y, _, _ = make_binary_linear(n=exp_dict["n_samples"],
                                        d=exp_dict["d"],
                                        margin=margin,
                                        y01=True,
                                        bias=True,
                                        separable=exp_dict.get("separable", True),
                                        seed=42)
        # No shuffling to keep the support vectors inside the training set
        splits = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
        X_train, X_test, Y_train, Y_test = splits

        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)

        Y_train = torch.LongTensor(Y_train)
        Y_test = torch.LongTensor(Y_test)

        if train_flag:
            dataset = torch.utils.data.TensorDataset(X_train, Y_train)
        else:
            dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    
    if dataset_name == "matrix_fac":
        fname = datadir + 'matrix_fac.pkl'
        if not os.path.exists(fname):
            data = generate_synthetic_matrix_factorization_data()
            ut.save_pkl(fname, data)

        A, y = ut.load_pkl(fname)

        X_train, X_test, y_train, y_test = train_test_split(A, y, test_size=0.2, random_state=9513451)

        training_set = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
        test_set = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float))

        if train_flag:
            dataset = training_set
        else:
            dataset = test_set

    return DatasetWrapper(dataset, split=split)

class DatasetWrapper:
    def __init__(self, dataset, split):
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, target = self.dataset[index]


        return {"images":data, 
                'labels':target, 
                'meta':{'indices':index}}

class NlpDatasetWrapper(Dataset):
    def __init__(self, corpus, data, bptt):
        self.corpus = corpus
        self.data = data
        self.bptt = bptt
    
    def __len__(self):
        return self.data.size(0) // self.bptt

    def __getitem__(self, index):
        data, targets = nlp_utils.get_batch(self.data, index*self.bptt, self.bptt)

        return {"images":data, 
                'labels':targets, 
                'meta':{'indices':index}}


def compute_max_eta_logistic_loss(X):
    s_list = np.ones(X.shape[0])*-1
    for i in tqdm.tqdm(range(X.shape[0])):
        x = X[i]
        # xx = np.outer(x, x)
        # e = np.linalg.eigh(xx)[0].max()
        e = np.linalg.norm(x)**2
        s = 1./(0.5*e + 2*1e-4)
        s_list[i] = s
    return s_list.max(), s_list.min()

def compute_max_eta_squared_loss(X):
    s_list = np.ones(X.shape[0])*-1
    for i in tqdm.tqdm(range(X.shape[0])):
        x = X[i]
        # xx = np.outer(x, x)
        # e = np.linalg.eigh(xx)[0].max()
        e = np.linalg.norm(x)**2
        s = 1./(2*e + 2*1e-4)
        s_list[i] = s
    return s_list.max(), s_list.min()
# ===========================================================
# matrix matrix_factorization
def generate_synthetic_matrix_factorization_data(xdim=6, ydim=10, nsamples=1000, A_condition_number=1e-10):
    """
    Generate a synthetic matrix factorization dataset as suggested by Ben Recht.
    See: https://github.com/benjamin-recht/shallow-linear-net/blob/master/TwoLayerLinearNets.ipynb.
    """
    Atrue = np.linspace(1, A_condition_number, ydim
       ).reshape(-1, 1) * np.random.rand(ydim, xdim)
    # the inputs
    X = np.random.randn(xdim, nsamples)
    # the y's to fit
    Ytrue = Atrue.dot(X)
    data = (X.T, Ytrue.T)

    return data

# ===========================================================
# Helpers
import os
import urllib

import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from torchvision.datasets import MNIST


LIBSVM_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
LIBSVM_DOWNLOAD_FN = {"rcv1"       : "rcv1_train.binary.bz2",
                      "mushrooms"  : "mushrooms",
                      "a1a"  : "a1a",
                      "a2a"  : "a2a",
                      "ijcnn"      : "ijcnn1.tr.bz2",
                      "w8a"        : "w8a"}


def load_mnist(data_dir):
    dataset = MNIST(data_dir, train=True, transform=None,
          target_transform=None, download=True)

    X, y = dataset.data.numpy(), dataset.targets.numpy()
    X = X / 255.
    X = X.reshape((X.shape[0], -1))
    return X, y


def load_libsvm(name, data_dir):
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    fn = LIBSVM_DOWNLOAD_FN[name]
    data_path = os.path.join(data_dir, fn)

    if not os.path.exists(data_path):
        url = urllib.parse.urljoin(LIBSVM_URL, fn)
        print("Downloading from %s" % url)
        urllib.request.urlretrieve(url, data_path)
        print("Download complete.")

    X, y = load_svmlight_file(data_path)
    return X, y


def make_binary_linear(n, d, margin, y01=False, bias=False, separable=True, scale=1, shuffle=True, seed=None):
    assert margin >= 0.

    if seed:
        np.random.seed(seed)

    labels = [-1, 1]

    # Generate support vectors that are 2 margins away from each other
    # that is also linearly separable by a homogeneous separator
    w = np.random.randn(d); w /= np.linalg.norm(w)
    # Now we have the normal vector of the separating hyperplane, generate
    # a random point on this plane, which should be orthogonal to w
    p = np.random.randn(d-1); l = (-p@w[:d-1])/w[-1]
    p = np.append(p, [l])

    # Now we take p as the starting point and move along the direction of w
    # by m and -m to obtain our support vectors
    v0 = p - margin*w
    v1 = p + margin*w
    yv = np.copy(labels)

    # Start generating points with rejection sampling
    X = []; y = []
    for i in range(n-2):
        s = scale if np.random.random() < 0.05 else 1

        label = np.random.choice(labels)
        # Generate a random point with mean at the center 
        xi = np.random.randn(d)
        xi = (xi / np.linalg.norm(xi))*s

        dist = xi@w
        while dist*label <= margin:
            u = v0-v1 if label == -1 else v1-v0
            u /= np.linalg.norm(u)
            xi = xi + u
            xi = (xi / np.linalg.norm(xi))*s
            dist = xi@w

        X.append(xi)
        y.append(label)

    X = np.array(X).astype(float); y = np.array(y)#.astype(float)

    if shuffle:
        ind = np.random.permutation(n-2)
        X = X[ind]; y = y[ind]

    # Put the support vectors at the beginning
    X = np.r_[np.array([v0, v1]), X]
    y = np.r_[np.array(yv), y]

    if separable:
        # Assert linear separability
        # Since we're supposed to interpolate, we should not regularize.
        clff = SVC(kernel="linear", gamma="auto", tol=1e-10, C=1e10)
        clff.fit(X, y)
        assert clff.score(X, y) == 1.0

        # Assert margin obtained is what we asked for
        w = clff.coef_.flatten()
        sv_margin = np.min(np.abs(clff.decision_function(X)/np.linalg.norm(w)))
        
        if np.abs(sv_margin - margin) >= 1e-4:
            print("Prescribed margin %.4f and actual margin %.4f differ (by %.4f)." % (margin, sv_margin, np.abs(sv_margin - margin)))

    else:
        flip_ind = np.random.choice(n, int(n*0.01))
        y[flip_ind] = -y[flip_ind]

    if y01:
        y[y==-1] = 0

    if bias:
        # TODO: Get rid of this later, bias should be handled internally,
        #       this is just for ease of implementation for the Hessian
        X = np.c_[np.ones(n), X]

    return X, y, w, (v0, v1)

def rbf_kernel(A, B, sigma):
    # func = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=True)
    # result = func(torch.from_numpy(A.toarray())[None], torch.from_numpy(B.toarray())[None])
   
    # np.square(metrics.pairwise.pairwise_distances(A.toarray(), B.toarray(), metric="euclidean"))
    
    
    # numpy version
    distsq = np.square(metrics.pairwise.pairwise_distances(A, B, metric="euclidean"))
    K = np.exp(-1 * distsq/(2*sigma**2))
    return K