import inspect

####### gla(glance) #######

### *variables       (Everything???)   Accepts any number of positional arguments(variables you want to see)


def gl(*variables):
    frame = inspect.currentframe().f_back
    variable_names = {
        id(v): name for name, v in frame.f_locals.items() if id(v) in map(id, variables)
    }

    # Function to detect dimensions (depth and elements per level) of nested lists or tuples
    def get_dimensions(item, level=0, dims=None):
        if dims is None:
            dims = {}

        if isinstance(item, (list, tuple)):
            # If the current level is not recorded yet, initialize it
            if level not in dims:
                dims[level] = 0
            dims[level] += len(item)

            # Recursively calculate dimensions for the nested elements
            for sub_item in item:
                get_dimensions(sub_item, level + 1, dims)

        return dims

    for var in variables:
        var_name = variable_names.get(id(var), "Unknown")
        print(f"Variable '{var_name}':")
        print(f"  Type: {type(var).__name__}")  # Cleaned output for the type

        # If the variable is a list or tuple, calculate its dimensions (depth and elements per level)
        if isinstance(var, (list, tuple)):
            dimensions = get_dimensions(var)
            print(f"  Dimensions (Depth: {len(dimensions)} levels): {dimensions}")

        # If the variable has a 'shape' attribute (like NumPy arrays or pandas DataFrames)
        elif hasattr(var, "shape"):
            print(f"  Shape: {var.shape}")

        else:
            print(f"  Shape: lemeow~")

        print(f"  Value: {var}\n")
