import torch
import torch.nn as nn
import catSNN
import catCuda
import catCpp

mv = torch.tensor([[0.5,0.5*1.01,0.5,0.5],[1,1,1,0]]).cuda()
#0.5 0.5+0.5*0.99 (0.5+0.5*0.99)*0.99
out = catCuda.getSpikes(mv, 1-0.001)
print(out)
"""
theta = 6 - 0.001
for tim in range(100):
    mv = torch.randn(100,128, 768,180,dtype=torch.float32).cuda()
    #print("final2")
    summed_mv = mv.sum(dim=-1)
    one_d_tensor = mv[99][127][767]
    cumulative_sum = torch.zeros_like(one_d_tensor)
    cumulative_sum_spike=torch.zeros_like(one_d_tensor)
    for i in range(one_d_tensor.size(0)):
        cumulative_sum[i] = cumulative_sum[i - 1] + one_d_tensor[i] if i > 0 else one_d_tensor[i]
        if cumulative_sum[i] > theta:
            cumulative_sum[i] -= theta
            cumulative_sum_spike[i] = 1
    #print(cumulative_sum)
    #print(cumulative_sum_spike)
    #print("final1")
    # 调用 getSpikes 函数
    out = catCuda.getSpikes(mv, theta)
    print(torch.sum(cumulative_sum_spike-out[99][127][767]))
"""