import torch 

def discriminatorMask(inputs1, inputs2):
    seg1 = inputs1[1]
    seg2 = inputs2[1]
    erase_mask = torch.eq(torch.argmax(seg1, dim=-3, keepdim=True), torch.argmax(seg2, dim=-3, keepdim=True)).float()
    inputs1 = [x * erase_mask for x in inputs1]
    inputs2 = [x * erase_mask for x in inputs2]
    return inputs1, inputs2, erase_mask

