import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, acc):
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count_ptr)
    acc_old = tl.load(A_ptrs)
    acc += tl.load(A_ptrs)
    tl.store(A_ptrs, acc)
    tl.atomic_xchg(Count_ptr, count + 1)
    tl.atomic_xchg(Lock_ptr, 0)
    return acc_old, count

@triton.jit
def wait_for_value(V_ptr, val):
    while tl.atomic_cas(V_ptr, val, val) != val:
        pass

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_stages=4, num_warps=4)
    ],
    key=['input_size', 'num_experts'],
    reset_to_zero=['C_ptr', 'C_Lock_ptr', 'C_Count_ptr']
)
@triton.jit
def sort_kernel(
        E_ptr, O_ptr, S_ptr,
        C_ptr, C_Lock_ptr, C_Count_ptr,
        num_experts: tl.constexpr,
        input_size: tl.constexpr,
        E_BLOCK_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):

    block_id = tl.program_id(0)
    total_updates = tl.num_programs(0)
    expert_ids = tl.arange(0, num_experts)

    in_idxs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    E_ptrs = E_ptr + in_idxs
    E = tl.load(E_ptrs)
    mask = E[:, None] == expert_ids[None, :]
    indicator = mask.to(tl.int32)
    
    # Add to global bincount and retrieve old accumulator
    acc, _ = locked_add(C_Lock_ptr, C_Count_ptr, C_ptr + expert_ids, tl.sum(indicator, axis=0))
    acc += tl.cumsum(indicator, axis=0)

    # Wait for all to report in
    wait_for_value(C_Count_ptr, total_updates)

    # Compute global bincount offset [0, exp0_count, ...]
    expert_counts = tl.load(C_ptr + expert_ids) 
    expert_offset = tl.cumsum(expert_counts, axis=0) - expert_counts
    acc += expert_offset[None, :]

    dest_idxs = tl.where(mask, acc - 1, -1)
    dest_idxs_flat = tl.max(dest_idxs, axis=1)

    tl.store(O_ptr + dest_idxs_flat, in_idxs)
    tl.store(S_ptr + dest_idxs_flat, E)


def sort_expert_idxs(expert_idxs, num_experts):
    input_size = expert_idxs.size(0)
    E_BLOCK_SIZE = triton.next_power_of_2(num_experts)

    bincount = torch.zeros(num_experts, dtype=torch.int32, device=expert_idxs.device)
    bincount_lock = torch.tensor(0, dtype=torch.int32, device=expert_idxs.device)
    bincount_count = torch.tensor(0, dtype=torch.int32, device=expert_idxs.device)

    sorted_idxs = torch.zeros_like(expert_idxs) - 1
    argsort_idxs = torch.zeros_like(expert_idxs) - 1

    grid = lambda meta: (triton.cdiv(input_size, meta['BLOCK_SIZE']),)
    sort_kernel[grid](
        expert_idxs,
        argsort_idxs,
        sorted_idxs,
        bincount, bincount_lock, bincount_count,
        num_experts,
        input_size,
        E_BLOCK_SIZE
    )

    return sorted_idxs, argsort_idxs, bincount

