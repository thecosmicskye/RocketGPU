import rocketgpu_py as _rs


class DeviceArray:
    def __init__(self, name, entry):
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
    return {name: DeviceArray(name, entry) for name, entry in sim.device_arrays().items()}


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


    def __init__(self, array_name, field, typestr, base_ptr, count, itemsize, offset, shape):
        import numpy as np

        elem = np.dtype(typestr)
        strides = [itemsize]
        for dim in reversed(tuple(shape)):
            strides.insert(1, elem.itemsize)
            if len(strides) > 2:
                strides[1] = strides[2] * dim
        full_shape = (count,) + tuple(shape)
        full_strides = tuple(strides[: len(full_shape)])
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
    itemsize = layout["itemsize"]
    assert itemsize == entry["itemsize"], (array_name, itemsize, entry["itemsize"])
    return {
        field: DeviceFieldArray(array_name, field, typestr, base_ptr, count, itemsize, offset, shape)
        for field, typestr, offset, shape in layout["fields"]
    }


def host_struct_array(sim, array_name):

    import ctypes

    import numpy as np

    entry = sim.device_arrays()[array_name]
    ptr = entry["__cuda_array_interface__"]["data"][0]
    nbytes = entry["itemsize"] * entry["count"]
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    return np.frombuffer(buf, dtype=state_dtype(array_name))
