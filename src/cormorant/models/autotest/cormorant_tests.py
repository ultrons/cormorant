import torch
from torch.utils.data import DataLoader

import logging

from cormorant.so3_lib import rotations as rot
from cormorant.so3_lib import SO3WignerD
from cormorant.data import collate_fn


def covariance_test(model, data, siamese=False):
    logging.info('Beginning covariance test!')
    targets_rotout, outputs_rotin = [], []

    if siamese:
        logging.info('Testing a siamese network.')
        device, dtype = data['positions1'].device, data['positions1'].dtype
    else:
        device, dtype = data['positions'].device, data['positions'].dtype

    D, R, _ = rot.gen_rot(model.maxl, device=device, dtype=dtype)

    D = SO3WignerD(D).to(model.device, model.dtype)

    data_rotout = data
    data_rotin = {key: val.clone() if torch.is_tensor(val) else None for key, val in data.items()}

    if siamese:
        data_rotin['positions1'] = rot.rotate_cart_vec(R, data_rotin['positions1'])
        data_rotin['positions2'] = rot.rotate_cart_vec(R, data_rotin['positions2'])
    else:
        data_rotin['positions'] = rot.rotate_cart_vec(R, data_rotin['positions'])

    outputs_rotout, reps_rotout, _ = model(data_rotout, covariance_test=True)
    outputs_rotin, reps_rotin, _ = model(data_rotin, covariance_test=True)

    invariance_test = (outputs_rotout - outputs_rotin).norm().item()

    reps_rotout = [reps.apply_wigner(D) for reps in reps_rotout]

    rep_diff = [(level_in - level_out).abs() for (level_in, level_out) in zip(reps_rotin, reps_rotout)]

    covariance_test_norm = [torch.tensor([diff.norm() for diff in diffs_lvl]) for diffs_lvl in rep_diff]
    covariance_test_mean = [torch.tensor([diff.mean() for diff in diffs_lvl]) for diffs_lvl in rep_diff]
    covariance_test_max = [torch.tensor([diff.max() for diff in diffs_lvl]) for diffs_lvl in rep_diff]

    covariance_test_max_all = max([max(lvl) for lvl in covariance_test_max]).item()

    logging.info('Rotation Invariance test: {:0.5g}'.format(invariance_test))
    logging.info('Largest deviation in covariance test : {:0.5g}'.format(covariance_test_max_all))

    # If the largest deviation in the covariance test is greater than 1e-5,
    # display l1 and l2 norms of each irrep along each level.
    if covariance_test_max_all > 1e-5:
        logging.warning('Largest deviation in covariance test {:0.5g} detected! Detailed summary:'.format(covariance_test_max_all))
        for lvl_idx, (lvl_norm, lvl_mean, lvl_max) in enumerate(zip(covariance_test_norm, covariance_test_mean, covariance_test_max)):
            for ell_idx, (ell_norm, ell_mean, ell_max) in enumerate(zip(lvl_norm, lvl_mean, lvl_max)):
                logging.warning('(lvl, ell) = ({}, {}) -> {:0.5g} (norm) {:0.5g} (mean) {:0.5g} (max)'\
                                .format(lvl_idx, ell_idx, ell_norm, ell_mean, ell_max))


def permutation_test(model, data, siamese=False):
    logging.info('Beginning permutation test!')

    data_noperm = data

    if siamese:

        logging.info('Testing a siamese network.')

        mask1 = data['atom_mask1']
        # Generate a list of indices for each molecule.
        perm1 = 1*torch.arange(mask1.shape[1]).expand(mask1.shape[0], -1)
        for idx in range(mask1.shape[0]):
            num_atoms1 = (mask1[idx, :].long()).sum()
            perm1[idx, :num_atoms1] = torch.randperm(num_atoms1)
        # apply the permutation
        apply_perm1 = lambda mat: torch.stack([mat[idx, p] for (idx, p) in enumerate(perm1)])
        assert((mask1 == apply_perm1(mask1)).all())
        data_perm = {key: apply_perm1(val) if torch.is_tensor(val) and val.dim() > 1 and key[-1]=='1' else val for key, val in data.items()}

        mask2 = data['atom_mask2']
        # Generate a list of indices for each molecule.
        perm2 = 1*torch.arange(mask2.shape[1]).expand(mask2.shape[0], -1)
        for idx in range(mask2.shape[0]):
            num_atoms2 = (mask2[idx, :].long()).sum()
            perm2[idx, :num_atoms2] = torch.randperm(num_atoms2)
        # apply the permutation
        apply_perm2 = lambda mat: torch.stack([mat[idx, p] for (idx, p) in enumerate(perm2)])
        assert((mask2 == apply_perm2(mask2)).all())
        data_perm = {key: apply_perm2(val) if torch.is_tensor(val) and val.dim() > 1 and key[-1]=='2' else val for key, val in data_perm.items()}

    else:

        mask = data['atom_mask']
        # Generate a list of indices for each molecule.
        # We will generate a permutation only for the atoms that exist (are not masked.)
        perm = 1*torch.arange(mask.shape[1]).expand(mask.shape[0], -1)
        for idx in range(mask.shape[0]):
            num_atoms = (mask[idx, :].long()).sum()
            perm[idx, :num_atoms] = torch.randperm(num_atoms)
        # apply the permutation
        apply_perm = lambda mat: torch.stack([mat[idx, p] for (idx, p) in enumerate(perm)])
        assert((mask == apply_perm(mask)).all())
        data_perm = {key: apply_perm(val) if torch.is_tensor(val) and val.dim() > 1 and key[-1]=='1' else val for key, val in data.items()}

    outputs_perm = model(data_perm)
    outputs_noperm = model(data_noperm)

    invariance_test = (outputs_perm - outputs_noperm).abs().max()

    logging.info('Permutation Invariance test: {}'.format(invariance_test))


def batch_test(model, data):
    logging.info('Beginning batch invariance test!')
    data_split = {key: val.unbind(dim=0) if (torch.is_tensor(val) and val.numel() > 1) else val for key, val in data.items()}
    data_split = [{key: val[idx].unsqueeze(0) if type(val) is tuple else val for key, val in data_split.items()} for idx in range(len(data['charges']))]

    outputs_split = torch.cat([model(data_sub) for data_sub in data_split])
    outputs_full = model(data)

    invariance_test = (outputs_split - outputs_full).abs().max()

    logging.info('Batch invariance test: {}'.format(invariance_test))


def cormorant_tests(model, dataloader, args, tests=['covariance'], charge_scale=1, siamese=False):
    if not args.test:
        logging.info("WARNING: network tests disabled!")
        return

    logging.info("Testing network for symmetries:")
    model.eval()

    data = next(iter(dataloader))

    covariance_test(model, data, siamese=siamese)
    permutation_test(model, data, siamese=siamese)
    if not siamese:
        batch_test(model, data)

    logging.info('Test complete!')

