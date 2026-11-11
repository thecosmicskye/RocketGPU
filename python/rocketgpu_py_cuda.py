import rocketgpu_py as _rs


class DeviceArray:
    def __init__(self, name, entry, owner=None):
        self._owner = owner
        self.name = name
        self.__cuda_array_interface__ = entry["__cuda_array_interface__"]
        self.itemsize = entry["itemsize"]
        self.count = entry["count"]

    @property
    def ptr(self):
        return self.__cuda_array_interface__["data"][0]

    @property
    def nbytes(self):
        return self.itemsize * self.count

    def __repr__(self):
        return f"DeviceArray({self.name}, count={self.count}, itemsize={self.itemsize}, ptr=0x{self.ptr:x})"


def device_arrays(sim):
    return {name: DeviceArray(name, entry, sim) for name, entry in sim.device_arrays().items()}


_TYPESTR_TO_NUMPY = None


def state_layout(name):
    return _rs.state_layouts()[name]


def state_dtype(name):

    import numpy as np

    layout = state_layout(name)
    names, formats, offsets = [], [], []
    for field, typestr, offset, shape in layout["fields"]:
        names.append(field)
        formats.append((np.dtype(typestr), tuple(shape)) if shape else np.dtype(typestr))
        offsets.append(offset)
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": layout["itemsize"]})


class DeviceFieldArray:


    def __init__(self, array_name, field, typestr, base_ptr, count, itemsize, offset, shape, owner=None, lanes=None):
        import numpy as np

        elem = np.dtype(typestr)
        inner = []
        stride = elem.itemsize
        for dim in reversed(tuple(shape)):
            inner.insert(0, stride)
            stride *= dim
        lead = (count,) if lanes is None else (count, lanes[0])
        lead_strides = (itemsize,) if lanes is None else (itemsize, lanes[1])
        full_shape = lead + tuple(shape)
        full_strides = lead_strides + tuple(inner)
        self._owner = owner
        self.name = f"{array_name}.{field}"
        self.dtype = elem
        self.shape = full_shape
        self.strides = full_strides
        self.__cuda_array_interface__ = {
            "data": (base_ptr + offset, False),
            "shape": full_shape,
            "typestr": typestr,
            "strides": full_strides,
            "version": 3,
        }

    def __repr__(self):
        return f"DeviceFieldArray({self.name}, shape={self.shape}, dtype={self.dtype}, strides={self.strides})"


def field_arrays(sim, array_name):
    entry = sim.device_arrays()[array_name]
    base_ptr = entry["__cuda_array_interface__"]["data"][0]
    count = entry["count"]
    layout = state_layout(array_name)
    if "device" in layout and _rs.is_cuda_build:
        dev = layout["device"]
        assert dev["tile_bytes"] == entry["itemsize"], (array_name, dev["tile_bytes"], entry["itemsize"])
        return {
            field: DeviceFieldArray(array_name, field, typestr, base_ptr, count, dev["tile_bytes"], offset, shape, sim,
                                    (dev["tile_cars"], lane_stride))
            for field, typestr, offset, lane_stride, shape in dev["fields"]
        }
    itemsize = layout["itemsize"]
    assert itemsize == entry["itemsize"], (array_name, itemsize, entry["itemsize"])
    return {
        field: DeviceFieldArray(array_name, field, typestr, base_ptr, count, itemsize, offset, shape, sim)
        for field, typestr, offset, shape in layout["fields"]
    }


def host_struct_array(sim, array_name):
    if _rs.is_cuda_build:
        raise RuntimeError("host_struct_array requires a CPU build")

    import ctypes

    import numpy as np

    entry = sim.device_arrays()[array_name]
    ptr = entry["__cuda_array_interface__"]["data"][0]
    nbytes = entry["itemsize"] * entry["count"]
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    buf._owner = sim
    return np.frombuffer(memoryview(buf).toreadonly(), dtype=state_dtype(array_name))
