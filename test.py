import torch

def concatenate_tensors(tensor1, tensor2, order1, order2):

    D1 = tensor1.shape[0]
    D2 = tensor2.shape[0]
    D = D1 + D2

    result_shape = (D,) + tensor1.shape[1:]
    result = torch.zeros(result_shape, dtype=tensor1.dtype)

    for i, idx in enumerate(order1):
        result[idx] = tensor1[i]
 
    for i, idx in enumerate(order2):
        result[idx] = tensor2[i]

    return result

torchA = torch.tensor([
    [1,1,1],
    [2,2,2],
    [3,3,3]
])

torchB = torch.tensor([
    [4,4,4],
    [5,5,5]
])

orderA = [0,1,3]
orderB = [2,4]

tensorC = concatenate_tensors(torchA, torchB, orderA, orderB)

print(tensorC)