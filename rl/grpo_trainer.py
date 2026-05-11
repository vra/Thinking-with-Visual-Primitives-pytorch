"""
Group Relative Policy Optimization (GRPO) trainer.

Reference: DeepSeek-V4 paper & DeepSeekMath GRPO.

Key ideas:
  - For each prompt, sample G rollouts from the current policy.
  - Compute reward for each rollout.
  - Advantage = reward - mean(reward within group).
  - Update policy to maximize log_prob * advantage.
  - Add KL penalty against a reference (SFT) model.
"""

from typing import List, Dict, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm


class GRPOTrainer:
    """
    Simplified GRPO trainer for visual primitive reasoning.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        tokenizer: AutoTokenizer,
        reward_fns: List[Callable[[str, Dict], float]],
        group_size: int = 4,
        kl_coeff: float = 0.04,
        clip_eps: float = 0.2,
        lr: float = 1e-6,
        device: str = "cuda",
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_fns = reward_fns
        self.group_size = group_size
        self.kl_coeff = kl_coeff
        self.clip_eps = clip_eps
        self.device = device

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            betas=(0.9, 0.999),
        )

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _generate_rollouts(
        self,
        pixel_values: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        **kwargs,
    ) -> List[str]:
        """Generate group_size rollouts for the given prompts.
        
        Qwen2-VL has a bug in get_rope_index when use_cache=False during generate
        (past_key_values handling causes input_ids/attention_mask shape mismatch).
        We work around by temporarily enabling cache and eval mode.
        """
        all_texts = []
        was_training = self.model.training
        original_use_cache = getattr(self.model.config, 'use_cache', None)
        
        self.model.eval()
        if hasattr(self.model, 'config'):
            self.model.config.use_cache = True
        
        try:
            for _ in range(self.group_size):
                outputs = self.model.generate(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
                # Decode only the new tokens
                new_tokens = outputs[:, input_ids.shape[1]:]
                texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
                all_texts.extend(texts)
        finally:
            if was_training:
                self.model.train()
            if original_use_cache is not None:
                self.model.config.use_cache = original_use_cache
        
        return all_texts

    @torch.no_grad()
    def _compute_rewards(
        self,
        texts: List[str],
        metadata_list: List[Dict],
    ) -> torch.Tensor:
        """Compute rewards for each rollout."""
        rewards = []
        for text, meta in zip(texts, metadata_list):
            r = 0.0
            for fn in self.reward_fns:
                r += fn(text, meta)
            rewards.append(r / len(self.reward_fns))
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _compute_log_probs(
        self,
        model,
        pixel_values,
        full_input_ids,
        attention_mask,
        prompt_len: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Compute per-token log probs for generated (response) tokens only."""
        outputs = model(
            pixel_values=pixel_values,
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
        targets = full_input_ids[:, 1:]     # (B, L-1)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        # Mask: only count response tokens (after prompt) and non-padding
        mask = attention_mask[:, 1:].float()
        if prompt_len > 0:
            mask[:, :prompt_len - 1] = 0.0  # mask out prompt tokens
        token_log_probs = token_log_probs * mask
        # Sum over sequence
        seq_log_probs = token_log_probs.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return seq_log_probs

    def train_step(
        self,
        pixel_values: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        metadata_list: List[Dict],
        max_new_tokens: int = 512,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Single GRPO training step.
        Returns dict with loss and reward stats.
        """
        batch_size = input_ids.shape[0]

        # 1. Generate rollouts
        rollouts = self._generate_rollouts(
            pixel_values, input_ids, attention_mask, max_new_tokens, **kwargs
        )

        # 2. Compute rewards
        # Repeat metadata for each group member
        metadata_repeated = []
        for meta in metadata_list:
            metadata_repeated.extend([meta] * self.group_size)
        rewards = self._compute_rewards(rollouts, metadata_repeated)

        # 3. Compute group-relative advantages
        rewards_grouped = rewards.view(batch_size, self.group_size)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1)

        # 4. Re-tokenize rollouts for policy gradient
        # Build full sequences: prompt + rollout
        rollout_tokens = self.tokenizer(
            rollouts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_new_tokens,
        )
        rollout_ids = rollout_tokens["input_ids"].to(self.device)
        rollout_mask = rollout_tokens["attention_mask"].to(self.device)

        # Concatenate prompt and rollout
        prompt_ids_repeated = input_ids.repeat_interleave(self.group_size, dim=0)
        prompt_mask_repeated = attention_mask.repeat_interleave(self.group_size, dim=0)

        full_ids = torch.cat([prompt_ids_repeated, rollout_ids], dim=1)
        full_mask = torch.cat([prompt_mask_repeated, rollout_mask], dim=1)

        prompt_len = input_ids.shape[1]

        # 5. Compute log probs for current and reference policy
        # Qwen2-VL: pixel_values/image_grid_thw are not batched per-sample,
        # so compute log probs per rollout (batch_size=1) for compatibility.
        seq_log_probs_list = []
        ref_seq_log_probs_list = []
        old_seq_log_probs_list = []
        for i in range(full_ids.shape[0]):
            lp = self._compute_log_probs(
                self.model, pixel_values, full_ids[i:i+1], full_mask[i:i+1],
                prompt_len=prompt_len, **kwargs
            )
            seq_log_probs_list.append(lp)
            with torch.no_grad():
                ref_lp = self._compute_log_probs(
                    self.ref_model, pixel_values, full_ids[i:i+1], full_mask[i:i+1],
                    prompt_len=prompt_len, **kwargs
                )
                ref_seq_log_probs_list.append(ref_lp)
                old_seq_log_probs_list.append(lp.detach())

        seq_log_probs = torch.cat(seq_log_probs_list)
        ref_seq_log_probs = torch.cat(ref_seq_log_probs_list)
        old_seq_log_probs = torch.cat(old_seq_log_probs_list)

        # 6. Compute KL penalty
        kl_div = seq_log_probs - ref_seq_log_probs  # approx KL

        # 7. Policy loss with clipped surrogate objective (PPO-style, per GRPO)
        log_ratio = seq_log_probs - old_seq_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        loss = -torch.min(surr1, surr2).mean() + self.kl_coeff * kl_div.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
        }

    def train_epoch(self, dataloader: DataLoader, max_new_tokens: int = 512):
        self.model.train()
        stats = {"loss": 0.0, "mean_reward": 0.0}
        count = 0
        pbar = tqdm(dataloader, desc="RL Training")
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            pixel_values = batch.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            metadata_list = batch.get("metadata", [{}] * input_ids.shape[0])

            # Collect extra kwargs for VL models (e.g., image_grid_thw)
            extra_kwargs = {k: v.to(self.device) for k, v in batch.items()
                            if isinstance(v, torch.Tensor) and k not in
                            ["input_ids", "attention_mask", "pixel_values", "labels", "metadata"]}

            step_stats = self.train_step(
                pixel_values, input_ids, attention_mask, metadata_list, max_new_tokens, **extra_kwargs
            )
            for k in stats:
                stats[k] += step_stats[k]
            count += 1
            pbar.set_postfix({k: f"{v/count:.4f}" for k, v in stats.items()})

        return {k: v / max(count, 1) for k, v in stats.items()}