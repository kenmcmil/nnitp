

# Given a 'flat' index into a tensor, return the element index.
# Here, `input_shape` is the shape of the tensor, and `unit` is the
# index to an element of the tensor flattened into a vector.


def unflatten_unit(input_shape,unit):
    unit = unit[0] if isinstance(unit,tuple) else unit
    res = tuple()
    while len(input_shape) > 0:
        input_shape,dim = input_shape[:-1],input_shape[-1]
        res = (unit%dim,) + res
        unit = unit//dim
    return res
