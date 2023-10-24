import logging
import time
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn, optim
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
from torch.nn.functional import normalize
import sys
import logging.config

logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default": {
            "format": "{levelname} {asctime} | {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": sys.stdout,
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
})


# Get root logger (all other loggers will be derived from this logger's
# properties)
logger = logging.getLogger()
logger.warning("I will output to terminal")  # No output in notebook, goes to terminal

# assuming only a single handler has been setup (seems
# to be default in notebook), set that handler to go to stdout.
# logger.handlers[0].stream = sys.stdout

logger.warning("FOO")  # Prints: WARNING:root:FOO
logger.info("hello")

# Other loggers derive from the root logger, so you can also do:
logger2 = logging.getLogger("logger2")
logger2.warning("BAR")  # Prints: WARNING:logger2:BAR


# Set hardware

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def GlobalContrastNormalization(tensor: torch.tensor, scale='l2'):
    assert scale in ('l1', 'l2')
    n_features = int(np.prod(tensor.shape))

    tensor = tensor - torch.mean(tensor)

    if (scale == 'l1'):
        tensor = tensor / torch.mean(torch.abs(tensor))

    if (scale == 'l2'):
        tensor = tensor / torch.sqrt(torch.sum(tensor ** 2) / n_features)

    return tensor

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return [idx for idx, label in enumerate(labels) if label in targets]

normal_class = 0

n_classes = 2
normal_classes = tuple([normal_class])
outlier_classes = list(range(0, 10))
outlier_classes.remove(normal_class)

min_max = [(-0.8826567065619495, 9.001545489292527),
           (-0.6661464580883915, 20.108062262467364),
           (-0.7820454743183202, 11.665100841080346),
           (-0.7645772083211267, 12.895051191467457),
           (-0.7253923114302238, 12.683235701611533),
           (-0.7698501867861425, 13.103278415430502),
           (-0.778418217980696, 10.457837397569108),
           (-0.7129780970522351, 12.057777597673047),
           (-0.8280402650205075, 10.581538445782988),
           (-0.7369959242164307, 10.697039838804978)]

transform = Compose([ToTensor(),
                     Lambda(lambda x: GlobalContrastNormalization(x, scale='l1')),
                     Normalize([min_max[normal_class][0]],
                               [min_max[normal_class][1] - min_max[normal_class][0]])])
target_transform = Lambda(lambda x: int(x in outlier_classes))

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
    target_transform=target_transform,
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
    target_transform=target_transform,
)

train_idx_normal = get_target_label_idx(training_data.train_labels.clone().data.cpu().numpy(), normal_classes)
train_data = Subset(training_data, train_idx_normal)


class MNIST_LeNet_AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder: Same as Deep Out-of-Context (OOC) network
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(nn.functional.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(nn.functional.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = nn.functional.interpolate(nn.functional.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = nn.functional.interpolate(nn.functional.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = nn.functional.interpolate(nn.functional.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)

        return x


class MNIST_LeNet_Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(nn.functional.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(nn.functional.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Configuration for Pretrain and Train
optimizer_name: str = 'adam'
lr: float = 0.001
n_epochs: int = 150
lr_milestones: tuple = ()
batch_size: int = 128
weight_decay: float = 1e-6
n_jobs_dataloader: int = 0

ae_net = MNIST_LeNet_AutoEncoder().to(device)
net = MNIST_LeNet_Network().to(device)


def AutoEncoder_PreTrain():
    logger = logging.getLogger()

    train_loader = DataLoader(train_data, batch_size, num_workers=n_jobs_dataloader)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ae_net.parameters(), lr=lr, weight_decay=weight_decay,
                                 amsgrad=(optimizer_name == 'amsgrad'))
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

    logger.info('Starting pretraining...')
    start_time = time.time()
    ae_net.train()

    # Training loop
    for epoch in range(n_epochs):
        schedular.step()
        if epoch in lr_milestones:
            logger.info('LR Scheduler: new learning rate is %g' % float(schedular.get_lr()[0]))
        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            # Zero the network parameters gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        epoch_train_time = time.time() - epoch_start_time
        logger.info('Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'.format(epoch + 1, n_epochs, epoch_train_time,
                                                                       loss_epoch / n_batches))

    pretrain_time = time.time() - start_time
    logger.info('Pretraining time: %.3f' % pretrain_time)
    logger.info('Finished pretraining.')


def AutoEncoder_Testing():
    from sklearn.metrics import roc_auc_score
    from sklearnex import patch_sklearn
    patch_sklearn()

    logger = logging.getLogger()

    # Get test data loader
    test_loader = DataLoader(test_data, batch_size, num_workers=n_jobs_dataloader)

    # Testing
    logger.info('Testing autoencoder...')
    loss_epoch = 0.0
    n_batches = 0
    start_time = time.time()
    label_score = []
    ae_net.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = ae_net(inputs)
            scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
            loss = torch.mean(scores)

            # Save triple of (idx, label, score) in a list
            label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                    scores.cpu().data.numpy().tolist()))

            loss_epoch += loss.item()
            n_batches += 1

    logger.info('Test set Loss: {:.8f}'.format(loss_epoch / n_batches))

    labels, scores = zip(*label_score)
    labels = np.array(labels)
    scores = np.array(scores)

    auc = roc_auc_score(labels, scores)
    logger.info('Test set AUC: {:.2f}%'.format(100. * auc))

    test_time = time.time() - start_time
    logger.info('Autoencoder testing time: %.3f' % test_time)
    logger.info('Finished testing autoencoder.')

# Save Pretrain model
def save_ae():
    ae_net_dict = ae_net.state_dict()
    torch.save({'ae_net_dict': ae_net_dict}, 'saved_model/ae_net.tar')

# Load Pretrain model
def load_ae():
    model_dict = torch.load('saved_model/ae_net.tar')
    ae_net.load_state_dict(model_dict['ae_net_dict'])

load_ae()
AutoEncoder_Testing()

# Init Network Weights from Pretraining
# def init_network_weights_from_pretraining():
"""Initialize the Deep SVDD network weights from the encoder weights of the pretraining autoencoder."""

net_dict = net.state_dict()
ae_net_dict = ae_net.state_dict()

# Filter out decoder network keys
ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
# Overwrite values in the existing state_dict
net_dict.update(ae_net_dict)
# Load the new state_dict
net.load_state_dict(net_dict)

# Init Hypersphere
R = 0.0     # Hypersphere radius R
c = None    # Hypersphere center c

R = torch.tensor(R, device=device)
# c = torch.tensor(c, device=device)
nu: float = 0.1
assert (0 < nu) & (nu <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."

warm_up_n_epochs = 10   # Number of training epochs for soft-boundary Deep SVDD before radius R gets updated

objective: str = 'one-class'
assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."

# Results
train_time = None
test_auc = None
test_time = None
test_scores = None

def init_center_c(train_loader: DataLoader, eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(net.rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)


# def train():
logger = logging.getLogger()

# Get train data loader
train_loader = DataLoader(train_data, batch_size, num_workers=n_jobs_dataloader)

# Set optimizer (Adam optimizer for now)
optimizer = torch.optim.Adam(ae_net.parameters(), lr=lr, weight_decay=weight_decay,
                             amsgrad=(optimizer_name == 'amsgrad'))

# Set learning rate scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

# Initialize hypersphere center c (if c not loaded)
if c is None:
    logger.info('Initializing center c...')
    c = init_center_c(train_loader)
    logger.info('Center c initialized.')

# Training
logger.info('Starting training...')
start_time = time.time()
net.train()
for epoch in range(n_epochs):

    scheduler.step()
    if epoch in lr_milestones:
        logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

    loss_epoch = 0.0
    n_batches = 0
    epoch_start_time = time.time()
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)

        # Zero the network parameter gradients
        optimizer.zero_grad()

        # Update network parameters via backpropagation: forward + backward + optimize
        outputs = net(inputs)
        dist = torch.sum((outputs - c) ** 2, dim=1)
        if objective == 'soft-boundary':
            scores = dist - R ** 2
            loss = R ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        else:
            loss = torch.mean(dist)
        loss.backward()
        optimizer.step()

        # Update hypersphere radius R on mini-batch distances
        if (objective == 'soft-boundary') and (epoch >= warm_up_n_epochs):
            R.data = torch.tensor(get_radius(dist, nu), device=device)

        loss_epoch += loss.item()
        n_batches += 1

    # log epoch statistics
    epoch_train_time = time.time() - epoch_start_time
    logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                .format(epoch + 1, n_epochs, epoch_train_time, loss_epoch / n_batches))

train_time = time.time() - start_time
logger.info('Training time: %.3f' % train_time)

logger.info('Finished training.')