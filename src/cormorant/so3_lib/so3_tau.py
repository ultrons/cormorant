import torch

from math import inf

from itertools import zip_longest

class SO3Tau():
    """
    Class for keeping track of multiplicity (number of channels) of a SO(3)
    vector.

    Parameters
    ----------

    tau : :obj:`list` of `int`, :obj:`SO3Tau`, or object with `.tau` property.
        Multiplicity of a SO(3) vector.
    """
    def __init__(self, tau):
        if type(tau) in [list, tuple]:
            if not all(type(t) == int for t in tau):
                raise ValueError('Input must be list or tuple of ints! {} {}'.format(type(tau), [type(t) for t in tau]))
        else:
            try:
                tau = tau.tau
            except e:
                raise ValueError('Input does not have a defined .tau property!')

        self._tau = tuple(tau)

    @property
    def maxl(self):
        return len(self._tau) - 1

    def keys(self):
        return range(len(self))

    def values(self):
        return self._tau

    def items(self):
        return zip(self._tau, range(len(self)))

    def __iter__(self):
        """
        Loop over SO3Tau
        """
        for t in self._tau:
            yield t

    def __getitem__(self, idx):
        """
        Get item of SO3Tau.
        """
        if type(idx) is slice:
            return SO3Tau(self._tau[idx])
        else:
            return self._tau[idx]

    def __len__(self):
        """
        Length of SO3Tau
        """
        return len(self._tau)

    def __setitem__(self, idx, val):
        """
        Set index of SO3Tau
        """
        self._tau[idx] = val

    def __eq__(self, other):
        """
        Check equality of two :math:`SO3Tau` objects or :math:`SO3Tau` and
        a list.
        """
        self_tau = tuple(self._tau)
        other_tau = tuple(other)

        return self_tau == other_tau

    @staticmethod
    def cat(tau_list):
        """
        Return the multiplicity :obj:`SO3Tau` corresponding to the concatenation
        (direct sum) of a list of :obj:`SO3Tensor`s.

        Parameters
        ----------
        tau_list : :obj:`list` of :obj:`SO3Tau` or :obj:`list` of :obj:`ints`
            List of multiplicites of input :obj:`SO3Tensors`.

        Return
        ------

        tau : :obj:`SO3Tau`
            Output tau of direct sum of input :obj:`SO3Tensor`s.

        """
        return SO3Tau([sum(taus) for taus in zip_longest(*tau_list, fillvalue=0)])


    def __and__(self, other):
        return SO3Tau.cat([self, other])

    def __rand__(self, other):
        return SO3Tau.cat([self, other])

    def __str__(self):
        return str(list(self._tau))

    __repr__ = __str__

    def __add__(self, other):
        return SO3Tau(list(self) + list(other))

    def __radd__(self, other):
        """
        Reverse add, includes type checker to deal with sum([])
        """
        if type(other) is int:
            return self
        return SO3Tau(list(other) + list(self))

    @staticmethod
    def from_rep(rep):
        """
        Construct SO3Tau object from an SO3Vector representation.

        Parameters
        ----------
        rep : :obj:`SO3Tensor` :obj:`list` of :obj:`torch.Tensors`
            Input representation.

        """
        from cormorant.so3_lib.so3_tensor import SO3Tensor

        if isinstance(SO3Tensor, rep):
            return rep.tau

        assert type(rep) is list and all(type(irrep) == torch.Tensor for irrep in rep), 'Input must be list of torch.Tensors! {} {}'.format(type(rep), [type(irrep) for irrep in rep])

        ells = [(irrep[0].shape[-2] - 1) // 2 for irrep in rep]

        minl, maxl = ells[0], ells[-1]

        assert ells == list(range(minl, maxl+1)), 'Rep must be continuous from minl to maxl'

        tau = [irrep.shape[-3] for irrep in rep]

        return SO3Tau(tau)

    @property
    def tau(self):
        return self._tau

    @property
    def channels(self):
        channels = set(self._tau)
        if len(channels) == 1:
            return channels.pop()
        else:
            return None
