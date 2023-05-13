import torch

a = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
print(a)
b = a.view(-1, 3, 4)
print(b)
print(b[0,1])
print(b.mean(dim=0, dtype=torch.float32))
print(b.shape)