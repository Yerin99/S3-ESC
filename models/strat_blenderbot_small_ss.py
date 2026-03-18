# coding=utf-8
"""
Strategy Blenderbot Small with Scheduled Sampling

This model extends strat_blenderbot_small with scheduled sampling
for strategy token prediction during training.
"""

import torch
import torch.nn.functional as F
from models.strat_blenderbot_small import Model as BaseModel
from transformers.models.blenderbot_small import BlenderbotSmallConfig


class Model(BaseModel):
    """
    Blenderbot Small with Strategy Scheduled Sampling.

    During training, instead of always using ground-truth strategy tokens,
    this model can use its own predictions or random strategies based on
    a schedule defined by StrategyScheduledSampler.
    """

    def __init__(self, config: BlenderbotSmallConfig):
        super().__init__(config)
        self.scheduled_sampler = None

    def set_scheduled_sampler(self, sampler):
        """Set the scheduled sampler for training."""
        self.scheduled_sampler = sampler

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        past_key_values=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        validation=False,
        **kwargs
    ):
        """
        Forward pass with optional scheduled sampling for strategy tokens.

        During training with scheduled sampling enabled:
        1. Encode input context
        2. Predict strategy logits using decoder with <s> token
        3. Sample strategy token based on schedule (GT, predicted, or random)
        4. Replace first token of decoder_input_ids with sampled strategy
        5. Continue with standard forward pass
        """
        assert self.toker is not None

        # Use scheduled sampling only during training
        use_scheduled_sampling = (
            self.training
            and not validation
            and self.scheduled_sampler is not None
            and decoder_input_ids is not None
        )

        if use_scheduled_sampling:
            p_gt, p_sample, p_random = self.scheduled_sampler.get_current_probs()
            use_scheduled_sampling = (p_sample > 0.0 or p_random > 0.0)

        if use_scheduled_sampling:
            # Step 1: Encode input if not already encoded
            return_dict_val = return_dict if return_dict is not None else self.config.use_return_dict

            if encoder_outputs is None:
                encoder_outputs = self.model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=return_dict_val,
                )

            # Step 2: Get strategy prediction using only <s> token
            # decoder_input_ids: [batch, seq_len] where first token is <s>, second is strategy
            bos_token_ids = decoder_input_ids[:, :1]  # [batch, 1]

            with torch.no_grad():
                decoder_outputs = self.model.decoder(
                    input_ids=bos_token_ids,
                    encoder_hidden_states=encoder_outputs[0],
                    encoder_attention_mask=attention_mask,
                    return_dict=return_dict_val,
                )
                strategy_logits = self.lm_head(decoder_outputs.last_hidden_state) + self.final_logits_bias
                # Extract logits for strategy tokens (last 8 positions in vocab)
                strategy_logits = strategy_logits[:, 0, -8:]  # [batch, 8]

            # Step 3: Get ground-truth strategy IDs (relative to strategy vocab, 0-7)
            # decoder_input_ids[:, 1] contains strategy token IDs (absolute vocab IDs)
            gt_strategy_token_ids = decoder_input_ids[:, 1]  # [batch]
            vocab_size = len(self.toker) - 8  # Base vocab size
            gt_strategy_ids = gt_strategy_token_ids - vocab_size  # Convert to 0-7 range

            # Step 4: Sample strategy tokens based on schedule
            sampled_strategy_ids = self.scheduled_sampler.sample_strategy_tokens(
                gt_strategy_ids=gt_strategy_ids,
                strategy_logits=strategy_logits,
            )

            # Step 5: Convert back to absolute vocab IDs and update decoder_input_ids
            sampled_strategy_token_ids = sampled_strategy_ids + vocab_size
            decoder_input_ids = decoder_input_ids.clone()
            decoder_input_ids[:, 1] = sampled_strategy_token_ids

            # Continue with standard forward pass using updated decoder_input_ids
            # Pass encoder_outputs to avoid recomputation
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                return_dict=return_dict,
                validation=validation,
                **kwargs
            )

        # Standard forward pass (no scheduled sampling)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
            validation=validation,
            **kwargs
        )
