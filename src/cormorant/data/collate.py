import torch


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """
    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (batch['charges'].sum(0) > 0)

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    batch['atom_mask'] = atom_mask
    batch['edge_mask'] = edge_mask

    return batch


def collate_siamese(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """

    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    assert batch['active'].shape[0] == 1

    # The following assumes a batch size of 1
    size_once = int(batch['charges'].shape[1]/2)
    charges1 = batch['charges'][:,:size_once]
    charges2 = batch['charges'][:,size_once:]
    to_keep1 = (charges1.sum(0) > 0)
    to_keep2 = (charges2.sum(0) > 0)

    new_batch = {}
    # Copy label data. TODO: Generalize to more labels and arbitrary label names!
    new_batch['label'] = batch['label']
    # Split structural data
    new_batch['charges1']   = drop_zeros( batch['charges'][:,:size_once], to_keep1 )
    new_batch['charges2']   = drop_zeros( batch['charges'][:,size_once:], to_keep2 )
    new_batch['positions1'] = drop_zeros( batch['positions'][:,:size_once,:], to_keep1 )
    new_batch['positions2'] = drop_zeros( batch['positions'][:,size_once:,:], to_keep2 )
    new_batch['one_hot1']   = drop_zeros( batch['one_hot'][:,:size_once,:], to_keep1 )
    new_batch['one_hot2']   = drop_zeros( batch['one_hot'][:,size_once:,:], to_keep2 )

    atom_mask1 = new_batch['charges1'] > 0
    atom_mask2 = new_batch['charges2'] > 0
    edge_mask1 = atom_mask1.unsqueeze(1) * atom_mask1.unsqueeze(2)
    edge_mask2 = atom_mask2.unsqueeze(1) * atom_mask2.unsqueeze(2)

    new_batch['atom_mask1'] = atom_mask1
    new_batch['atom_mask2'] = atom_mask2
    new_batch['edge_mask1'] = edge_mask1
    new_batch['edge_mask2'] = edge_mask2

    return new_batch


def collate_activity(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """

    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    assert batch['active'].shape[0] == 1

    # The following assumes a batch size of 1 and 
    # that all atoms of the active structure are saved first
    size_once = int(batch['active'].sum())
    charges1 = batch['charges'][:,:size_once]
    charges2 = batch['charges'][:,size_once:]
    to_keep1 = (charges1.sum(0) > 0)
    to_keep2 = (charges2.sum(0) > 0)

    new_batch = {}
    # Copy label data. TODO: Generalize to more labels and arbitrary label names!
    new_batch['label'] = batch['label']
    # Split structural data
    new_batch['charges1']   = drop_zeros( batch['charges'][:,:size_once], to_keep1 )
    new_batch['charges2']   = drop_zeros( batch['charges'][:,size_once:], to_keep2 )
    new_batch['positions1'] = drop_zeros( batch['positions'][:,:size_once,:], to_keep1 )
    new_batch['positions2'] = drop_zeros( batch['positions'][:,size_once:,:], to_keep2 )
    new_batch['one_hot1']   = drop_zeros( batch['one_hot'][:,:size_once,:], to_keep1 )
    new_batch['one_hot2']   = drop_zeros( batch['one_hot'][:,size_once:,:], to_keep2 )

    atom_mask1 = new_batch['charges1'] > 0
    atom_mask2 = new_batch['charges2'] > 0
    edge_mask1 = atom_mask1.unsqueeze(1) * atom_mask1.unsqueeze(2)
    edge_mask2 = atom_mask2.unsqueeze(1) * atom_mask2.unsqueeze(2)

    new_batch['atom_mask1'] = atom_mask1
    new_batch['atom_mask2'] = atom_mask2
    new_batch['edge_mask1'] = edge_mask1
    new_batch['edge_mask2'] = edge_mask2

    return new_batch

