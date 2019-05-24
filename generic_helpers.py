
def print_dict_keys(print_dict, a=0, ident=2, max_depth=100):
    """Prints all keys of a dictionary recursively up until the maximum depth.
    
    Arguments:
        print_dict {dict} -- The python dictionary to print.
    
    Keyword Arguments:
        a {int} -- Counter for the recursive calling of the print function. (default: {0})
        ident {int} -- The number of spaces to use to ident the different key levels. (default: {2})
        max_depth {int} -- The maximum depth in the print_dict to print the keys. (default: {100})
    """
    for i, (key, value) in enumerate(print_dict.items()):
        print(" "*a + f"[{key}]")

        if isinstance(value, dict) and max_depth > a/ident:
            print_dict_keys(value, a+ident, max_depth=max_depth)