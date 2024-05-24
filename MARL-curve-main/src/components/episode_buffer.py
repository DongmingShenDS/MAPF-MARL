# DS DONE 1
import torch as th
import numpy as np
from types import SimpleNamespace as SN


class EpisodeBatch:
    """EpisodeBatch class in components/transforms"""

    def __init__(self,
                 scheme,
                 groups,
                 batch_size,
                 max_seq_length,
                 data=None,  # DS: should be of SN (SimpleNamespace)
                 preprocess=None,
                 device="cpu"):
        """Init"""
        self.scheme = scheme.copy()  # DS: scheme are dictionary of informations
        self.groups = groups  # DS: number of agents
        self.batch_size = batch_size  # DS: batch_size, usually 32
        self.max_seq_length = max_seq_length  # DS: max episode len + 1
        self.preprocess = {} if preprocess is None else preprocess  # DS: preprocess onehot actions
        self.device = device  
        # DS: if given data, assign to data
        if data is not None:
            self.data = data
        # DS: otherwise, init data as SN (SimpleNamespace) ~ better dict
        else:
            self.data = SN()
            self.data.transition_data = {}
            self.data.episode_data = {}
            # DS: initilize the data => _setup_data
            self._setup_data(self.scheme, self.groups, batch_size, max_seq_length, self.preprocess)

    def _setup_data(self, scheme, groups, batch_size, max_seq_length, preprocess):
        """Setup self.data"""
        # DS: preprocess ? TODO
        if preprocess is not None:
            for k in preprocess:
                assert k in scheme
                new_k = preprocess[k][0]
                transforms = preprocess[k][1]
                vshape = self.scheme[k]["vshape"]
                dtype = self.scheme[k]["dtype"]
                for transform in transforms:
                    vshape, dtype = transform.infer_output_info(vshape, dtype)
                self.scheme[new_k] = {
                    "vshape": vshape,
                    "dtype": dtype
                }
                if "group" in self.scheme[k]:
                    self.scheme[new_k]["group"] = self.scheme[k]["group"]
                if "episode_const" in self.scheme[k]:
                    self.scheme[new_k]["episode_const"] = self.scheme[k]["episode_const"]
        # DS: ? TODO
        assert "filled" not in scheme, '"filled" is a reserved key for masking.'
        scheme.update({
            "filled": {"vshape": (1,), "dtype": th.long},
        })
        # DS: iterate through all dict key/info pairs in scheme
        # DS: field_key includes: state, obs, actions, avail_actions, reward, terminated
        for field_key, field_info in scheme.items():
            assert "vshape" in field_info, "Scheme must define vshape for {}".format(field_key)
            vshape = field_info["vshape"]
            episode_const = field_info.get("episode_const", False)
            group = field_info.get("group", None)
            dtype = field_info.get("dtype", th.float32)
            # DS: vshape setup
            if isinstance(vshape, int):
                vshape = (vshape,)
            # DS: shape setup (from group)
            if group:
                assert group in groups, "Group {} must have its number of members defined in _groups_".format(group)
                shape = (groups[group], *vshape)
            else:
                shape = vshape
            # DS: if episode_const, then set up episode_data = store episode data in buffer
            if episode_const:
                self.data.episode_data[field_key] = th.zeros((batch_size, *shape), dtype=dtype, device=self.device)
            # DS: otherwise, then set up transition_data = store transition data in buffer
            else:  # DS: usually we have transition_data instead of episode_data
                self.data.transition_data[field_key] = th.zeros(
                    (batch_size, max_seq_length, *shape), 
                    dtype=dtype, device=self.device
                )

    def extend(self, scheme, groups=None):
        # DS: missing self.preprocess at the end of _setup_data(...) ? TODO
        self._setup_data(scheme, self.groups if groups is None else groups, self.batch_size, self.max_seq_length)

    def to(self, device):
        """Transfer everything to device (cpu/cuda)"""
        for k, v in self.data.transition_data.items():
            self.data.transition_data[k] = v.to(device)
        for k, v in self.data.episode_data.items():
            self.data.episode_data[k] = v.to(device)
        self.device = device

    def update(self, data, bs=slice(None), ts=slice(None), mark_filled=True):
        # DS: data, bs, ts, mark_filled ? TODO
        """Update contents in EpisodeBatch"""
        # DS: parse (bs, ts) to slices with => _parse_slices(...)
        slices = self._parse_slices((bs, ts))
        # DS: iterate through all items (as dict k/v pairs) in data
        for k, v in data.items():
            # DS: key in data is transition_data, setup target & get _slices
            if k in self.data.transition_data:
                target = self.data.transition_data
                if mark_filled:
                    target["filled"][slices] = 1
                    mark_filled = False
                _slices = slices
            # DS: key in data is episode_data, setup target & get _slices
            elif k in self.data.episode_data:
                target = self.data.episode_data
                _slices = slices[0]
            # DS: error elements type in data
            else:
                raise KeyError("{} not found in transition or episode data".format(k))
            # DS: set up v (value)'s dtype according to scheme
            dtype = self.scheme[k].get("dtype", th.float32)
            v = th.tensor(v, dtype=dtype, device=self.device)
            # DS: check if the update is safe => _check_safe_view(...). new = v; old/slot = target[k][_slices]
            # DS: if safe, do the update
            self._check_safe_view(v, target[k][_slices])
            target[k][_slices] = v.view_as(target[k][_slices])
            # DS: preprocess ? TODO
            if k in self.preprocess:
                new_k = self.preprocess[k][0]
                v = target[k][_slices]  # v now are regular actions
                for transform in self.preprocess[k][1]:  # DS: this step One-hot the actions
                    v = transform.transform(v)  # v now are one-hot encoded actions
                target[new_k][_slices] = v.view_as(target[new_k][_slices])

    def _check_safe_view(self, v, dest):
        # v = new data, dest = destination
        """Check if adding v to dest is safe"""
        # from IPython import embed; embed()  # breakpoint
        idx = len(v.shape) - 1
        for s in dest.shape[::-1]:
            if v.shape[idx] != s:
                if s != 1:
                    raise ValueError("Unsafe reshape of {} to {}".format(v.shape, dest.shape))
            else:
                idx -= 1

    def __getitem__(self, item):
        """Redefine __getitem__(...) for getter"""
        # DS: if item type is str
        if isinstance(item, str):
            # DS: simply return the corresponding item based on types (episode_data/transition_data)
            if item in self.data.episode_data:
                return self.data.episode_data[item]
            elif item in self.data.transition_data:
                return self.data.transition_data[item]
            else:
                raise ValueError
        # DS: if item type is tuple of strs
        elif isinstance(item, tuple) and all([isinstance(it, str) for it in item]):
            # DS: create SN structure called new_data to hold the data
            new_data = self._new_data_sn()
            # DS: iteratively add items based on types into the new_data
            for key in item:
                if key in self.data.transition_data:
                    new_data.transition_data[key] = self.data.transition_data[key]
                elif key in self.data.episode_data:
                    new_data.episode_data[key] = self.data.episode_data[key]
                else:
                    raise KeyError("Unrecognised key {}".format(key))
            # Update the scheme to only have the requested keys
            new_scheme = {key: self.scheme[key] for key in item}
            new_groups = {self.scheme[key]["group"]: self.groups[self.scheme[key]["group"]]
                          for key in item if "group" in self.scheme[key]}
            # DS: return the result new EpisodeBatch
            ret = EpisodeBatch(new_scheme, new_groups, self.batch_size, self.max_seq_length, data=new_data,
                               device=self.device)
            return ret
        # DS: all other item types
        else:
            # DS: parse item & create SN structure for new_data => _parse_slices & _new_data_sn
            item = self._parse_slices(item)
            new_data = self._new_data_sn()
            # DS: iteratively add items based on types into the new_data
            for k, v in self.data.transition_data.items():
                new_data.transition_data[k] = v[item]
            for k, v in self.data.episode_data.items():
                new_data.episode_data[k] = v[item[0]]
            # DS: get sizes of item ? => _get_num_items(...) TODO
            ret_bs = self._get_num_items(item[0], self.batch_size)
            ret_max_t = self._get_num_items(item[1], self.max_seq_length)
            # DS: return the result new EpisodeBatch
            ret = EpisodeBatch(self.scheme, self.groups, ret_bs, ret_max_t, data=new_data, device=self.device)
            return ret

    def _get_num_items(self, indexing_item, max_size):
        """Get the number of items in indexing_item"""
        if isinstance(indexing_item, list) or isinstance(indexing_item, np.ndarray):
            return len(indexing_item)
        elif isinstance(indexing_item, slice):
            _range = indexing_item.indices(max_size)
            return 1 + (_range[1] - _range[0] - 1) // _range[2]

    def _new_data_sn(self):
        """Create an SN (SimpleNamespace) holder for new_data"""
        new_data = SN()
        new_data.transition_data = {}
        new_data.episode_data = {}
        return new_data

    def _parse_slices(self, items):
        parsed = []
        # Only batch slice given, add full time slice
        if (isinstance(items, slice)  # slice a:b
                or isinstance(items, int)  # int i
                or (isinstance(items, (list, np.ndarray, th.LongTensor, th.cuda.LongTensor)))  # [a,b,c]
        ):
            # DS: items[0]=original items, items[1]=new slice
            # DS: python slice() = https://www.w3schools.com/python/ref_func_slice.asp
            items = (items, slice(None))
        # Need the time indexing to be contiguous
        if isinstance(items[1], list):
            raise IndexError("Indexing across Time must be contiguous")
        # DS: parse through items & attach to parsed
        for item in items:
            # TODO: stronger checks to ensure only supported options get through
            if isinstance(item, int):
                # Convert single indices to slices
                parsed.append(slice(item, item + 1))
            else:
                # Leave slices and lists as is
                parsed.append(item)
        return parsed

    def max_t_filled(self):
        # where this is called? TODO
        return th.sum(self.data.transition_data["filled"], 1).max(0)[0]

    def __repr__(self):
        """Redefine __repr__(...) for output message"""
        return "EpisodeBatch. Batch Size:{} Max_seq_len:{} Keys:{} Groups:{}".format(self.batch_size,
                                                                                     self.max_seq_length,
                                                                                     self.scheme.keys(),
                                                                                     self.groups.keys())


class ReplayBuffer(EpisodeBatch):
    # Called in run.py => buffer=ReplayBuffer(...)
    """ReplayBuffer class in components/transforms, inherit from EpisodeBatch class"""

    def __init__(self, scheme, groups, buffer_size, max_seq_length, preprocess=None, device="cpu"):
        super(ReplayBuffer, self).__init__(scheme, groups, buffer_size, max_seq_length, preprocess=preprocess,
                                           device=device)
        # DS: same as self.batch_size but more explicit
        self.buffer_size = buffer_size  # DS: buffer size, usually 5000
        self.buffer_index = 0
        # DS: keep track of the number of episodes in the buffer
        self.episodes_in_buffer = 0

    def insert_episode_batch(self, ep_batch):
        if self.buffer_index + ep_batch.batch_size <= self.buffer_size:
            self.update(ep_batch.data.transition_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size),
                        slice(0, ep_batch.max_seq_length),
                        mark_filled=False)
            self.update(ep_batch.data.episode_data,
                        slice(self.buffer_index, self.buffer_index + ep_batch.batch_size))
            self.buffer_index = (self.buffer_index + ep_batch.batch_size)
            self.episodes_in_buffer = max(self.episodes_in_buffer, self.buffer_index)
            self.buffer_index = self.buffer_index % self.buffer_size
            assert self.buffer_index < self.buffer_size
        else:
            buffer_left = self.buffer_size - self.buffer_index
            self.insert_episode_batch(ep_batch[0:buffer_left, :])
            self.insert_episode_batch(ep_batch[buffer_left:, :])

    def can_sample(self, batch_size):
        """Return can_sample=True: if episodes_in_buffer >= batch_size"""
        return self.episodes_in_buffer >= batch_size

    def sample(self, batch_size):
        """Sample batch_size from the buffer"""
        assert self.can_sample(batch_size)
        if self.episodes_in_buffer == batch_size:
            return self[:batch_size]
        else:
            # Uniform sampling only atm
            ep_ids = np.random.choice(self.episodes_in_buffer, batch_size, replace=False)
            return self[ep_ids]

    def __repr__(self):
        return "ReplayBuffer. {}/{} episodes. Keys:{} Groups:{}".format(self.episodes_in_buffer,
                                                                        self.buffer_size,
                                                                        self.scheme.keys(),
                                                                        self.groups.keys())
