import torch.nn as nn

#note: ai generated. seems to work ok
def wrap_sub_module(model, path, wrapper):
    """
    Wraps a submodule in the model as specified by the path with the given wrapper.

    Parameters:
    - model: the PyTorch model/module object
    - path: list of attributes (or indices) to navigate the model hierarchy
    - wrapper: a function that takes a module and returns its wrapped version

    Returns:
    - the model with the specified submodule wrapped
    """

    # Recursive function to navigate through the path
    def _replace_module(model, path):
        # Base case: If we're at the end of the path, wrap the module
        if len(path) == 1:
            module = getattr(model, path[0])
            setattr(model, path[0], wrapper(module))
            return

        # Otherwise, get the next attribute/module in the path
        next_module = getattr(model, path[0])

        # If the next attribute is a ModuleList or ModuleDict and the next path item is an int or str,
        # then get the specific module by index or key, respectively
        if isinstance(next_module, nn.ModuleList) and isinstance(path[1], int):
            next_module = next_module[path[1]]
            _replace_module(next_module, path[2:])
        elif isinstance(next_module, nn.ModuleDict) and isinstance(path[1], str):
            next_module = next_module[path[1]]
            _replace_module(next_module, path[2:])
        else:
            _replace_module(next_module, path[1:])

    _replace_module(model, path)
    return model

# # Example of use:

# class ExampleModel(nn.Module):
#     def __init__(self):
#         super(ExampleModel, self).__init__()
#         self.h = nn.ModuleList([
#             nn.ModuleDict({
#                 "attn": nn.Linear(10, 10)
#             }),
#             nn.ModuleDict({
#                 "attn": nn.Linear(10, 10)
#             })
#         ])

# # Function to wrap the module
# def my_wrapper(module):
#     return nn.Sequential(module, nn.ReLU())

# model = ExampleModel()
# print(model)

# model = wrap_sub_module(model, ["h", 0, "attn"], my_wrapper)
# print(model)
