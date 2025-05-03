import os
import inspect
import sympy as sp
from colorama import init, Fore, Style

init(autoreset=True)

####### gla(glance) #######

### *variables       (Everything???)   Accepts any number of positional arguments(variables you want to see)
### sa(see all)      bool              whether see all element


def gl(*variables, sa=True):
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
        if sa:
            print(f"  Value:                                                   {var}\n")


####### vfs(view file system) #######


### path             str               The path of the part you wanna know the structure
### pr               str               The prefix used for indentation (used internally during recursion)
### f                str | list[str]   Filter keyword(s) to apply on file/folder names
### m                str               Mode of filtering: "kp" to *keep* matching entries, "ig" to *ignore* them
### depth            int | None        Max recursion depth (None means unlimited)


def vfs(path, pr="", f=None, m="ig", depth=None):
    """Print a tree structure from the given path, showing full path only at the root.
    Folders are suffixed with '/' for clarity.
    Filtering supports 'kp' or 'ig' logic.
    """

    if depth is not None and depth <= 0:
        print(Fore.RED + f"{pr}[!] Invalid depth value: {depth} — must be > 0 or None")
        return

    def matches(name):
        if not f:
            return True
        filters = [f] if isinstance(f, str) else f
        return any(f in name for f in filters)

    def should_include(name):
        if f is None:
            return True
        return matches(name) if m == "kp" else not matches(name)

    def dir_has_match(p, remaining_depth):
        """Check if dir or any child matches filter within depth."""
        if remaining_depth == 0:
            return should_include(os.path.basename(p))
        try:
            for root, dirs, files in os.walk(p):
                rel_depth = root[len(p) :].count(os.sep)
                if rel_depth > remaining_depth:
                    break
                if should_include(os.path.basename(root)):
                    return True
                for name in dirs + files:
                    if should_include(name):
                        return True
        except:
            pass
        return False

    if not os.path.exists(path):
        print(f"{pr}[!] Path does not exist: {path}")
        return

    if not pr:
        print(os.path.abspath(path))  # Print root once

    try:
        raw_entries = sorted(os.listdir(path))
    except PermissionError:
        print(pr + "[!] Permission denied")
        return

    # Apply filtering
    entries = []
    for entry in raw_entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            if should_include(entry) or dir_has_match(
                full_path, None if depth is None else depth - 1
            ):
                entries.append(entry)
        elif should_include(entry):
            entries.append(entry)

    entries_count = len(entries)

    for index, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        is_dir = os.path.isdir(full_path)
        is_last = index == entries_count - 1
        connector = "└── " if is_last else "├── "
        name = entry + "/" if is_dir else entry
        print(pr + connector + name)

        if is_dir and (depth is None or depth > 1):
            extension = "    " if is_last else "│   "
            vfs(full_path, pr + extension, f, m, None if depth is None else depth - 1)
