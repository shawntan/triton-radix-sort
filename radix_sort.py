import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def locked_add(Lock_ptr, Count_ptr, A_ptrs, acc, mask, NO_MASK):
    while tl.atomic_cas(Lock_ptr, 0, 1) == 1:
        pass
    count = tl.load(Count_ptr)

    if NO_MASK:
        acc_old = tl.load(A_ptrs)
    else:
        acc_old = tl.load(A_ptrs, mask=mask)

    acc_new = acc + acc_old
    tl.store(A_ptrs, acc_new, mask=mask)

    new_count = count + 1
    tl.atomic_xchg(Count_ptr, new_count)
    tl.atomic_xchg(Lock_ptr, 0)
    return acc_new, acc_old, new_count

@triton.jit
def wait_for_value(V_ptr, val):
    while tl.atomic_cas(V_ptr, val, val) != val:
        pass

def get_configs():
    configs = [
        triton.Config({'BLOCK_SIZE': 2**i, 'INNER_BLOCK_SIZE': 2**j},
                      num_stages=num_stages, num_warps=num_warps)
        for i in range(10, 13)
        for j in range(8, i + 1)
        for num_stages in [2, 4]
        for num_warps in [4, 8]
        if i <= j
    ]
    return configs

@triton.autotune(
    configs=get_configs(),
    # configs=[triton.Config({'BLOCK_SIZE': 4096, 'INNER_BLOCK_SIZE': 1024}, num_stages=4, num_warps=8)],
    key=['input_size', 'num_experts'],
    reset_to_zero=['C_ptr', 'C_Lock_ptr', 'C_Count_ptr'],
)
@triton.heuristics(
    values={
        'E_BLOCK_SIZE': lambda META: triton.next_power_of_2(META['num_experts']),
        'NO_E_MASK': lambda META: triton.next_power_of_2(META['num_experts']) == META['num_experts'],
        'IS_MONOBLOCK': lambda META: META['BLOCK_SIZE'] == META['INNER_BLOCK_SIZE'],
    }
)
@triton.jit
def sort_kernel(
        E_ptr, O_ptr, S_ptr,
        C_ptr, C_Lock_ptr, C_Count_ptr,
        num_experts: tl.constexpr,
        input_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        INNER_BLOCK_SIZE: tl.constexpr,
        E_BLOCK_SIZE: tl.constexpr,
        NO_E_MASK: tl.constexpr,
        IS_MONOBLOCK: tl.constexpr
    ):

    block_id = tl.program_id(0)
    total_updates = tl.num_programs(0)
    expert_ids = tl.arange(0, E_BLOCK_SIZE)
    bincount_mask = expert_ids < num_experts

    if IS_MONOBLOCK:
        NO_INPUT_MASK: tl.constexpr = input_size % BLOCK_SIZE == 0
        sort_kernel_monoblock(
            block_id, total_updates, expert_ids, bincount_mask,
            E_ptr, O_ptr, S_ptr,
            C_ptr, C_Lock_ptr, C_Count_ptr,
            num_experts,
            input_size,
            BLOCK_SIZE,
            E_BLOCK_SIZE, NO_E_MASK,
            NO_INPUT_MASK=NO_INPUT_MASK and (block_id == tl.num_programs(0) - 1)
        )
    else:
        NO_INPUT_MASK: tl.constexpr = input_size % INNER_BLOCK_SIZE == 0
        sort_kernel_multiblock(
            block_id, total_updates, expert_ids, bincount_mask,
            E_ptr, O_ptr, S_ptr,
            C_ptr, C_Lock_ptr, C_Count_ptr,
            num_experts,
            input_size,
            BLOCK_SIZE,
            INNER_BLOCK_SIZE,
            E_BLOCK_SIZE, NO_E_MASK,
            NO_INPUT_MASK=NO_INPUT_MASK and (block_id == tl.num_programs(0) - 1)
        )


@triton.jit
def sort_kernel_monoblock(
        block_id, total_updates, expert_ids, bincount_mask,
        E_ptr, O_ptr, S_ptr,
        C_ptr, C_Lock_ptr, C_Count_ptr,
        num_experts: tl.constexpr,
        input_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        E_BLOCK_SIZE: tl.constexpr,
        NO_E_MASK: tl.constexpr,
        NO_INPUT_MASK: tl.constexpr
    ):

    in_idxs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    E_ptrs = E_ptr + in_idxs
    if NO_INPUT_MASK:
        E = tl.load(E_ptrs)
    else:
        input_mask = in_idxs < input_size
        E = tl.load(E_ptrs, mask=input_mask, other=num_experts)

    mask = E[:, None] == expert_ids[None, :]
    indicator = tl.where(mask, 1, 0)

    # Add to global bincount and retrieve old accumulator

    expert_counts, acc, update_counts = locked_add(
        C_Lock_ptr, C_Count_ptr, C_ptr + expert_ids, tl.sum(indicator, axis=0),
        mask=bincount_mask, NO_MASK=NO_E_MASK
    )
    acc += tl.cumsum(indicator, axis=0) # do something before waiting...
    if update_counts < total_updates:
        # Wait for all to report in
        wait_for_value(C_Count_ptr, total_updates)
        # Compute global bincount offset [0, exp0_count, ...]
        expert_counts = tl.load(C_ptr + expert_ids, mask=bincount_mask) 

    expert_offset = tl.cumsum(expert_counts, axis=0) - expert_counts
    acc += expert_offset[None, :]
    dest_idxs = tl.where(mask, acc - 1, -1)
    dest_idxs_flat = tl.max(dest_idxs, axis=1)

    if NO_INPUT_MASK:
        tl.store(O_ptr + dest_idxs_flat, in_idxs)
        tl.store(S_ptr + dest_idxs_flat, E)
    else:
        input_mask = in_idxs < input_size
        tl.store(O_ptr + dest_idxs_flat, in_idxs, mask=input_mask)
        tl.store(S_ptr + dest_idxs_flat, E, mask=input_mask)


@triton.jit
def sort_kernel_multiblock(
        block_id, total_updates, expert_ids, bincount_mask,
        E_ptr, O_ptr, S_ptr,
        C_ptr, C_Lock_ptr, C_Count_ptr,
        num_experts: tl.constexpr,
        input_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        INNER_BLOCK_SIZE: tl.constexpr,
        E_BLOCK_SIZE: tl.constexpr,
        NO_E_MASK: tl.constexpr,
        NO_INPUT_MASK: tl.constexpr):

    operating_size = tl.minimum(BLOCK_SIZE, input_size - block_id * BLOCK_SIZE)
    iters = tl.cdiv(operating_size, INNER_BLOCK_SIZE)
    in_idxs = block_id * BLOCK_SIZE + tl.arange(0, INNER_BLOCK_SIZE)
    E_ptrs = E_ptr + in_idxs

    acc_block_counts = tl.zeros_like(expert_ids)
    for inner_block_id in tl.range(iters - 1):
        E = tl.load(E_ptrs)
        mask = E[:, None] == expert_ids[None, :]
        indicator = tl.where(mask, 1, 0)
        acc_block_counts += tl.sum(indicator, axis=0)
        E_ptrs += INNER_BLOCK_SIZE
        in_idxs += INNER_BLOCK_SIZE

    if NO_INPUT_MASK:
        E = tl.load(E_ptrs)
    else:
        input_mask = in_idxs < input_size
        E = tl.load(E_ptrs, mask=input_mask, other=num_experts)

    mask = E[:, None] == expert_ids[None, :]
    indicator = tl.where(mask, 1, 0)
    acc_block_counts += tl.sum(indicator, axis=0)

    # Add to global bincount and retrieve old accumulator
    expert_counts, prev_counts, update_counts = locked_add(
        C_Lock_ptr, C_Count_ptr, C_ptr + expert_ids, acc_block_counts,
        mask=bincount_mask, NO_MASK=NO_E_MASK
    )

    # Reset pointers while waiting for all to report in
    in_idxs = block_id * BLOCK_SIZE + tl.arange(0, INNER_BLOCK_SIZE) # do something before waiting...
    E_ptrs = E_ptr + in_idxs

    if update_counts < total_updates: # Check if last thread to report.
        wait_for_value(C_Count_ptr, total_updates)
        # Compute global bincount offset [0, exp0_count, ...]
        expert_counts = tl.load(C_ptr + expert_ids, mask=bincount_mask) 

    expert_offset = tl.cumsum(expert_counts, axis=0) - expert_counts + prev_counts

    for inner_block_id in tl.range(iters - 1):
        E = tl.load(E_ptrs)
        mask = E[:, None] == expert_ids[None, :]
        indicator = tl.where(mask, 1, 0)
        ptrs = expert_offset + tl.cumsum(indicator, axis=0) - 1
        ptrs = tl.where(mask, ptrs, -1)
        ptrs = tl.max(ptrs, axis=1)

        tl.store(O_ptr + ptrs, in_idxs)
        tl.store(S_ptr + ptrs, E)

        expert_offset += tl.sum(indicator, axis=0)
        E_ptrs += INNER_BLOCK_SIZE
        in_idxs += INNER_BLOCK_SIZE

    E = tl.load(E_ptrs)
    mask = E[:, None] == expert_ids[None, :]
    indicator = tl.where(mask, 1, 0)
    ptrs = expert_offset + tl.cumsum(indicator, axis=0) - 1
    ptrs = tl.where(mask, ptrs, -1)
    ptrs = tl.max(ptrs, axis=1)

    if NO_INPUT_MASK:
        tl.store(O_ptr + ptrs, in_idxs)
        tl.store(S_ptr + ptrs, E)
    else:
        input_mask = in_idxs < input_size
        tl.store(O_ptr + ptrs, in_idxs, mask=input_mask)
        tl.store(S_ptr + ptrs, E, mask=input_mask)



 
def sort_expert_idxs(expert_idxs, num_experts):
    input_size = expert_idxs.size(0)
    bincount = torch.zeros(num_experts, dtype=torch.int32, device=expert_idxs.device)
    bincount_lock = torch.tensor(0, dtype=torch.int32, device=expert_idxs.device)
    bincount_count = torch.tensor(0, dtype=torch.int32, device=expert_idxs.device)
    sorted_idxs = torch.zeros_like(expert_idxs) - 1
    argsort_idxs = torch.zeros_like(expert_idxs) - 1

    grid = lambda meta: (triton.cdiv(input_size, meta['BLOCK_SIZE']),)
    sort_kernel[grid](
        expert_idxs,
        argsort_idxs, sorted_idxs,
        bincount, bincount_lock, bincount_count,
        num_experts, input_size,
    )
    return sorted_idxs, argsort_idxs, bincount

