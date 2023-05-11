# Ed Callaghan
# General, helpful, functions
# February 2021

def BuildObject(obj, defaults, overrides):
    obj.__dict__.update(defaults)
    obj.__dict__.update(overrides)
    return obj
