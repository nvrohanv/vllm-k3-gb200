"""Does vLLM's fused_grouped_topk match DeepSeekV3-style routing (1 group) for K3?"""
import torch
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import fused_grouped_topk

torch.manual_seed(0)
E, K, M = 896, 16, 4096
logits = torch.randn(M, E, device="cuda") * 2.0
bias = (torch.randn(E, device="cuda") * 0.5).float()
w, ids = fused_grouped_topk(hidden_states=torch.empty(M, 7168, device="cuda", dtype=torch.bfloat16),
                            gating_output=logits, topk=K, renormalize=True,
                            e_score_correction_bias=bias, num_expert_group=1, topk_group=1,
                            scoring_func="sigmoid", routed_scaling_factor=1.0)
s = logits.sigmoid()
ref_ids = torch.topk(s + bias, K, dim=-1).indices
ref_w = s.gather(1, ref_ids)
ref_w = ref_w / ref_w.sum(-1, keepdim=True)
same_set = (torch.sort(ids.long(), -1).values == torch.sort(ref_ids, -1).values).all(-1).float().mean().item()
# weights aligned by id
wd = torch.zeros(M, E, device="cuda").scatter(1, ids.long(), w.float())
rd = torch.zeros(M, E, device="cuda").scatter(1, ref_ids, ref_w)
print(f"expert-set agreement {same_set:.4%}, max |w diff| {(wd - rd).abs().max().item():.2e}, dtypes {w.dtype} {ids.dtype}")
print("weights sum:", w.float().sum(-1)[:4].tolist())
