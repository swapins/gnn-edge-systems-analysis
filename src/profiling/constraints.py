def enforce_constraints(memory, max_memory=None):
    if max_memory is None:
        return True

    if memory > max_memory:
        raise RuntimeError(f"Memory exceeded: {memory} > {max_memory}")

    return True