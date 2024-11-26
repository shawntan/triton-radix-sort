import torch
from radix_sort import sort_expert_idxs

if __name__ == "__main__":
    num_experts = 14
    expert_idxs = torch.randint(num_experts, (12345,)) # 
    expert_idxs = expert_idxs.to(torch.device('cuda:0'))
    print("Input:")
    print(expert_idxs)
    sorted_expert_idxs, argsort_idxs, count = sort_expert_idxs(expert_idxs, num_experts)

    print("Output:")
    assert (count == torch.bincount(expert_idxs)).all()
    print(count)
    print(torch.bincount(expert_idxs))
    assert (sorted_expert_idxs[:-1] <= sorted_expert_idxs[1:]).all()
    print(sorted_expert_idxs)
    assert (sorted_expert_idxs == torch.sort(expert_idxs)[0]).all()