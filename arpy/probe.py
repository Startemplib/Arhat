import inspect
import sympy as sp

####### gla(glance) #######


### *variables       (Everything???)   Accepts any number of positional arguments(variables you want to see)
# Test the type and dimensions (depth and elements per level) of variables
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

    def analyze_sympy_expression(expr, element_types):
        if isinstance(expr, sp.Basic):
            if expr.is_Add or expr.is_Mul or expr.is_Pow:
                element_types.add(type(expr).__name__)

                # Decompose the Add, Mul, or Pow into its arguments
                for arg in expr.args:
                    if isinstance(arg, sp.Basic):
                        element_types.add(type(arg).__name__)  # Add sub-expression type
                        analyze_sympy_expression(
                            arg, element_types
                        )  # Recursively analyze the arguments
            else:
                # Collect the symbols involved in the expression
                for symbol in expr.free_symbols:
                    element_types.add(f"Symbol({symbol})")

    # Function to collect all basic element types within lists, tuples, or SymPy matrices
    def get_element_types(item, element_types=None):
        if element_types is None:
            element_types = set()

            # Handle SymPy expressions (Add, Mul, symbols, etc.)
        if isinstance(
            item, (sp.Matrix, sp.ImmutableDenseMatrix, sp.MutableDenseMatrix)
        ):
            for element in item:
                analyze_sympy_expression(element, element_types)

        # Check for SymPy symbols
        elif isinstance(
            item, sp.Basic
        ):  # This includes symbolic elements like sp.symbols
            element_types.add(type(item).__name__)

        elif isinstance(item, (list, tuple)):
            # Recursively get types of nested elements
            for sub_item in item:
                get_element_types(sub_item, element_types)
        else:
            # Add the type of the basic element
            element_types.add(type(item).__name__)

        return element_types

    for var in variables:
        var_name = variable_names.get(id(var), "Unknown")
        print(f"Name:                                                    '{var_name}'")
        print(
            f"  Type:                                                    {type(var).__name__}"
        )  # Cleaned output for the type

        # If the variable is a list or tuple, calculate its dimensions and element types
        if isinstance(var, (list, tuple)):
            dimensions = get_dimensions(var)
            element_types = get_element_types(var)
            print(
                f"  Dimensions (Depth: {len(dimensions)}):                       {dimensions}"
            )
            print(
                f"  Element Types:                                    {element_types}"
            )

        # If the variable has a 'shape' attribute (like NumPy arrays or SymPy matrices), show shape
        elif hasattr(var, "shape"):
            print(
                f"  Shape:                                                   {var.shape}"
            )
            element_types = get_element_types(var)
            print(
                f"  Element Types:                                    {element_types}"
            )

        else:
            print(f"  Shape:                                                   lemeow~")

        print(f"  Value:                                                   {var}\n")
