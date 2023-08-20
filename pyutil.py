import torch

def print_tensor_attributes(tensor):
    # Get all attributes of the tensor
    attributes = dir(tensor)

    # Print all attributes and values
    for attr in attributes:
        try:
            attribute = getattr(tensor, attr)
            if not callable(attribute):
                print(f'{attr}: {attribute}')
            else:
                value = attribute()
                print(f'{attr}: {value}')
        except Exception as e:
            1
                #print(f'{attr}: ERROR ({str(e)})')
# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])

# Print all attributes and values of the tensor
print_tensor_attributes(x)
