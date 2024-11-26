import torch
import triton
from transformers import set_seed
from radix_sort import sort_expert_idxs


def ref_fwd(expert_idxs, E):
    sorted_expert_ids, argsort = torch.sort(expert_idxs)
    counts = torch.bincount(sorted_expert_ids, minlength=E)
    return sorted_expert_ids, argsort, counts

def tri_fwd(expert_idxs, E):
    sorted_expert_idxs, argsort, counts = sort_expert_idxs(expert_idxs, E)
    return sorted_expert_idxs, argsort, counts
    


providers = [
    ("reference", "torch.sort + torch.bincount", ("red", "-")),
    ("triton", "triton-radix-sort", ("blue", "-")),
]
@triton.testing.perf_report([
    triton.testing.Benchmark(
        x_names=["length"],
        x_vals=[2**(i  + 12) for i in range(6)],
        line_arg="provider",
        line_vals=[x[0] for x in providers],
        line_names=[x[1] for x in providers],
        styles=[x[2] for x in providers],
        ylabel="ms",
        plot_name=f"triton v torch",
        args={}
    )
])
def benchmark_varlen(length, provider):
    warmup = 100
    rep = 1000
    device = torch.device('cuda:0')
    num_experts = 8
    expert_idxs = torch.randint(num_experts, (length,), device=device, dtype=torch.int32) 

    set_seed(1337)
    if provider == "triton":
        fun = lambda: tri_fwd(expert_idxs, E=num_experts)
    else:
        fun = lambda: ref_fwd(expert_idxs, E=num_experts)


    return triton.testing.do_bench(fun, warmup=warmup, rep=rep)



if __name__ == "__main__":
    benchmark_varlen.run(save_path=None, print_data=True)
