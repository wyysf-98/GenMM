"""
this file borrows some codes from https://github.com/ariel415el/Efficient-GPNN/blob/main/utils/NN.py.
"""
import torch
import torch.nn.functional as F
import unfoldNd

def extract_patches(x, patch_size, stride, loop=False):
    """Extract patches from a motion sequence"""
    b, c, _t = x.shape

    # manually padding to loop
    if loop:
        half = patch_size // 2
        front, tail = x[:,:,:half], x[:,:,-half:]
        x = torch.concat([tail, x, front], dim=-1)

    x_patches = unfoldNd.unfoldNd(x, kernel_size=patch_size, stride=stride).transpose(1, 2).reshape(b, -1, c, patch_size)
    
    return x_patches.view(b, -1, c * patch_size)

def combine_patches(x_shape, ys, patch_size, stride, loop=False):
    """Combine motion patches"""
    
    # manually handle the loop situation
    out_shape = [*x_shape]
    if loop:
        padding = patch_size // 2
        out_shape[-1] = out_shape[-1] + padding * 2

    combined = unfoldNd.foldNd(ys.permute(0, 2, 1), output_size=tuple(out_shape[-1:]), kernel_size=patch_size, stride=stride)

    # normal fold matrix
    input_ones = torch.ones(tuple(out_shape), dtype=ys.dtype, device=ys.device)
    divisor = unfoldNd.unfoldNd(input_ones, kernel_size=patch_size, stride=stride)
    divisor = unfoldNd.foldNd(divisor, output_size=tuple(out_shape[-1:]), kernel_size=patch_size, stride=stride)
    combined = (combined / divisor).squeeze(dim=0).unsqueeze(0)
    
    if loop:
        half = patch_size // 2
        front, tail = combined[:,:,:half], combined[:,:,-half:]
        combined[:, :, half:2 * half] = (combined[:, :, half:2 * half] + tail) / 2
        combined[:, :, - 2 * half:-half] = (front + combined[:, :, - 2 * half:-half]) / 2
        combined = combined[:, :, half:-half]

    return combined


def efficient_cdist(X, Y):
    """
    borrowed from https://github.com/ariel415el/Efficient-GPNN/blob/main/utils/NN.py
    Pytorch efficient way of computing distances between all vectors in X and Y, i.e (X[:, None] - Y[None, :])**2
    Get the nearest neighbor index from Y for each X
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices
    """
    dist = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * torch.mm(X, torch.transpose(Y, 0, 1))
    d = X.shape[1]
    dist /= d # normalize by size of vector to make dists independent of the size of d ( use same alpha for all patche-sizes)
    return dist # DO NOT use torch.sqrt


def get_col_mins_efficient(dist_fn, X, Y, b=1024):
    """
    borrowed from https://github.com/ariel415el/Efficient-GPNN/blob/main/utils/NN.py
    Computes the l2 distance to the closest x or each y.
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns n1 long array of L2 distances
    """
    n_batches = len(Y) // b
    mins = torch.zeros(Y.shape[0], dtype=X.dtype, device=X.device)
    for i in range(n_batches):
        mins[i * b:(i + 1) * b] = dist_fn(X, Y[i * b:(i + 1) * b]).min(0)[0]
    if len(Y) % b != 0:
        mins[n_batches * b:] = dist_fn(X, Y[n_batches * b:]).min(0)[0]

    return mins


def get_NNs_Dists(dist_fn, X, Y, alpha=None, b=1024):
    """
    borrowed from https://github.com/ariel415el/Efficient-GPNN/blob/main/utils/NN.py
    Get the nearest neighbor index from Y for each X.
    Avoids holding a (n1 * n2) amtrix in order to reducing memory footprint to (b * max(n1,n2)).
    :param X:  (n1, d) tensor
    :param Y:  (n2, d) tensor
    Returns a n2 n1 of indices amd distances
    """
    if alpha is not None:
        normalizing_row = get_col_mins_efficient(dist_fn, X, Y, b=b)
        normalizing_row = alpha + normalizing_row[None, :]
    else:
        normalizing_row = 1

    NNs = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    Dists = torch.zeros(X.shape[0], dtype=torch.float, device=X.device)

    n_batches = len(X) // b
    for i in range(n_batches):
        dists = dist_fn(X[i * b:(i + 1) * b], Y) / normalizing_row
        NNs[i * b:(i + 1) * b] = dists.min(1)[1]
        Dists[i * b:(i + 1) * b] = dists.min(1)[0]
    if len(X) % b != 0:
        dists = dist_fn(X[n_batches * b:], Y) / normalizing_row
        NNs[n_batches * b:] = dists.min(1)[1]
        Dists[n_batches * b: ] = dists.min(1)[0]

    return NNs, Dists
