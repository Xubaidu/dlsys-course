import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    return nn.Sequential(
        nn.Residual(modules),
        nn.ReLU(),
    )
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
    ]
    '''
    We must generate the block within each iteration to ensure the randomness of the parameters of linear module.
    The code below will fix the randomness which is unexpected and hard to debug:
    
    ```py
    block = ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob)
    for i in range(num_blocks):
        modules.append(block)
    ```
    '''
    for i in range(num_blocks):
        block = ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob)
        modules.append(block)
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    hit, loss = 0, 0
    loss_func = nn.SoftmaxLoss()
    if opt != None:
        model.train()
        for i, batch in enumerate(dataloader):
            # X and y are both Tensor type
            X, y = batch[0], batch[1]

            # forward, compute the loss
            output = model(X)
            l = loss_func(output, y)
            loss += l.numpy() * y.shape[0]

            # backward, compute the grad
            opt.reset_grad()
            l.backward()
            opt.step()

            # calculate the hit num
            hit += (y.numpy() == output.numpy().argmax(axis=1)).sum()
    else:
        model.eval()
        for i, batch in enumerate(dataloader):
            X, y = batch[0], batch[1]

            # forward, compute the loss
            output = model(X)
            l = loss_func(output, y)
            loss += l.numpy() * y.shape[0]

            # calculate the hit num
            hit += (y.numpy() == output.numpy().argmax(axis=1)).sum()

    whole_batch_size = len(dataloader.dataset)
    acc = float(1 - hit / whole_batch_size)
    loss = float(loss / whole_batch_size)
    return acc, loss

    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    mnist_train_dataset = ndl.data.MNISTDataset(f'{data_dir}/train-images-idx3-ubyte.gz',
                                                    f'{data_dir}/train-labels-idx1-ubyte.gz')
    mnist_test_dataset = ndl.data.MNISTDataset(f'{data_dir}/t10k-images-idx3-ubyte.gz',
                                                    f'{data_dir}/t10k-labels-idx1-ubyte.gz')

    train_dataloader = ndl.data.DataLoader(dataset=mnist_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(dataset=mnist_test_dataset, batch_size=batch_size, shuffle=True)

    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(
        lr=lr, params=model.parameters(),
        weight_decay=weight_decay) if optimizer is not None else None
    for i in range(epochs):
        train_acc, train_loss = epoch(train_dataloader, model, opt=opt)
    test_acc, test_loss = epoch(test_dataloader, model, opt=None)

    return train_acc, train_loss, test_acc, test_loss

    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
