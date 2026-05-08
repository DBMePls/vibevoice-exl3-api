from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm
import torch
import time
import torch.nn as nn

from transformers.models.auto import AutoModel, AutoModelForCausalLM

from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers import modeling_utils
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from exllamav3.ext import exllamav3_ext as ext # <--- ExLlamaV3 Extension imported here!
from exllamav3 import Config, Model, Cache, Tokenizer, DefaultSampler
from exllamav3.modules.attn import prepare_flash_attn
from exllamav3.util import Timer

from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache, VibeVoiceTokenizerEncoderOutput
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead
from schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice import VibeVoiceConfig

from .modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizer, VibeVoiceTextTokenizerFast
from .modeling_vibevoice import VibeVoiceModel, VibeVoicePreTrainedModel
from .streamer import AudioStreamer, AsyncAudioStreamer

logger = logging.get_logger(__name__)

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]

class ExLlamaV3Wrapper:
    """Wrapper class for ExLlamaV3 integration"""
    
    def __init__(self, model, positive_cache, negative_cache, config):
        self.model = model
        self.positive_cache = positive_cache
        self.negative_cache = negative_cache
        self.config = config
        self.hidden_size = config.hidden_size
        self.model.last_hidden_states = None
        self.positive_past_len = 0
        self.negative_past_len = 0
        self.base_params = {
            "attn_mode": "flash_attn",
            "batch_shape": (1, 2048),
        }
    
    def prefill(self, input_ids):
        self.model.prefill(
            input_ids=input_ids,
            params={
                "attn_mode": "flash_attn",
                "cache": self.cache,
                "past_len": 0,
                "batch_shape": (1, 2048),
            }
        )
    
    def get_input_embeddings(self):
        return self.model.modules[0]
    
    def compute_inputs_embeds(self, input_ids):
        embedding_module = self.get_input_embeddings()
        params = self.base_params.copy()
        params["past_len"] = 0 
        input_ids = embedding_module.prepare_for_device(input_ids, params)
        inputs_embeds = embedding_module.forward(input_ids, params)
        if hasattr(embedding_module, 'normalize') and embedding_module.normalize:
            inputs_embeds = inputs_embeds * (inputs_embeds.shape[-1] ** 0.5)
        return inputs_embeds
        
    def forward(self, input_ids=None, inputs_embeds=None, position_ids=None, use_negative_cache=False):
        if use_negative_cache:
            cache = self.negative_cache
            past_len = self.negative_past_len
        else:
            cache = self.positive_cache
            past_len = self.positive_past_len
            
        params = self.base_params.copy()
        params["cache"] = cache
        params["past_len"] = past_len
        if position_ids is not None:
            params["position_ids"] = position_ids.to(torch.int)
        
        if inputs_embeds is not None:
            params["seq_len"] = inputs_embeds.shape[1]
            device = self.model.modules[0].device
            if inputs_embeds.device != device:
                inputs_embeds = inputs_embeds.to(device)
            logits, hidden_states = self.model.forward(
                input_ids=None,
                params=params,
                inputs_embeds=inputs_embeds,
            )
        else:
            logits, hidden_states = self.model.forward(
                input_ids=input_ids,
                params=params,
                inputs_embeds=None,
            )
        
        if use_negative_cache:
            self.negative_past_len += input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        else:
            self.positive_past_len += input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            
        return logits, hidden_states
        
    def sample(self, logits, tokenizer):
        return self.sampler.forward(logits, tokenizer=tokenizer)
        
    def reset_cache(self, use_negative_cache=False, max_num_tokens=2048):
        if use_negative_cache:
            self.negative_cache = Cache(self.model, max_num_tokens=2048)
            self.negative_past_len = 0
        else:
            self.positive_cache = Cache(self.model, max_num_tokens=2048)
            self.positive_past_len = 0    

    def reset_all(self, max_num_tokens=2048):
        self.positive_past_len = 0
        self.negative_past_len = 0        
    
    def unload(self):
        try:
            if hasattr(self.model, 'unload'):
                self.model.unload()
            del self.positive_cache
            del self.negative_cache
            del self.model
            self.positive_cache = None
            self.negative_cache = None
            self.model = None
            self.positive_past_len = 0
            self.negative_past_len = 0
            import gc
            gc.collect()
            logger.info("ExLlamaV3 model unloaded successfully")
        except Exception as e:
            logger.error(f"Error unloading ExLlamaV3: {e}")    
        
@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None

@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None
    reach_max_step_sample: Optional[torch.BoolTensor] = None

class VibeVoiceTokenConstraintProcessor(LogitsProcessor):
    def __init__(self, valid_token_ids: List[int], device: torch.device = None):
        self.valid_token_ids = torch.tensor(valid_token_ids, dtype=torch.long, device=device)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        mask[:, self.valid_token_ids] = 0
        scores = scores + mask
        return scores
    
class VibeVoiceForConditionalGenerationInference(VibeVoicePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceModel(config)
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps
        self.exllama = None
        self.negative_llm_steps_to_cache = 0
        self.negative_outputs_stored = None
        self.increase_cfg = False           
        self._cpp_automaton = None # <--- Our C++ Extension Worker Tracker
        self.post_init()        

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head
    
    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def semantic_tokenizer(self):
        return self.model.semantic_tokenizer
    
    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    @property
    def semantic_connector(self):
        return self.model.semantic_connector

    def _get_cpp_automaton(self):
        """Lazily initialize the C++ Diffusion Automaton"""
        if self._cpp_automaton is not None:
            return self._cpp_automaton
            
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        head = self.model.prediction_head
        
        # Unpack layers
        norm_w, ffn_gate_w, ffn_up_w, ffn_down_w, adaln_w = [], [], [], [], []
        for layer in head.layers:
            norm_w.append(layer.norm.weight)
            ffn_gate_w.append(layer.ffn.gate_proj.weight)
            ffn_up_w.append(layer.ffn.up_proj.weight)
            ffn_down_w.append(layer.ffn.down_proj.weight)
            adaln_w.append(layer.adaLN_modulation[1].weight)
            
        # Get scheduler arrays safely
        device = head.noisy_images_proj.weight.device
        alpha_t = self.model.noise_scheduler.alpha_t.to(device).to(torch.float32)
        sigma_t = self.model.noise_scheduler.sigma_t.to(device).to(torch.float32)
        lambda_t = self.model.noise_scheduler.lambda_t.to(device).to(torch.float32)
        timesteps_list = self.model.noise_scheduler.timesteps.tolist()

        self._cpp_automaton = ext.VibeVoiceDiffusionWorker(
            head.noisy_images_proj.weight,
            head.cond_proj.weight,
            head.t_embedder.mlp[0].weight,
            head.t_embedder.mlp[2].weight,
            head.final_layer.linear.weight,
            head.final_layer.adaLN_modulation[1].weight,
            norm_w, ffn_gate_w, ffn_up_w, ffn_down_w, adaln_w,
            self.model.acoustic_connector.fc1.weight, self.model.acoustic_connector.fc1.bias,
            self.model.acoustic_connector.norm.weight, 
            self.model.acoustic_connector.fc2.weight, self.model.acoustic_connector.fc2.bias,
            alpha_t, sigma_t, lambda_t, timesteps_list,
            1e-5 # eps
        )
        return self._cpp_automaton
        
    def tie_weights(self):
        if not getattr(self.config, 'tie_word_embeddings', False):
            return
        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)
    
    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        with torch.no_grad():
            if speech_type == "audio":
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * self.model.speech_scaling_factor
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks]
                return acoustic_features, acoustic_connected
            elif speech_type == "pt":
                encoder_output = VibeVoiceTokenizerEncoderOutput(mean=speech_tensors, std=self.acoustic_tokenizer.config.fix_std)
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                acoustic_features = (acoustic_latents + self.model.speech_bias_factor.to(acoustic_latents.device)) * self.model.speech_scaling_factor
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks]
                return acoustic_features, acoustic_connected
            else:
                raise NotImplementedError(f"Speech type {speech_type} not implemented")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        use_exllama: Optional[bool] = False,
        past_len: Optional[int] = None, 
        use_negative_cache: Optional[bool] = False, 
        **kwargs,
    ) -> Union[Tuple, VibeVoiceCausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict        
        logits_need_squeezing = False
        
        if use_exllama and self.exllama is not None:
            if inputs_embeds is None:
                inputs_embeds = self.exllama.compute_inputs_embeds(input_ids)
                logits_need_squeezing = True
            
            if speech_tensors is not None and speech_masks is not None:
                acoustic_features, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
                if speech_input_mask is not None:   
                    inputs_embeds = inputs_embeds.to(self.device)
                    inputs_embeds[speech_input_mask] = speech_embeds.float().to(self.device) 
            
            if inputs_embeds is not None:
                logits, hidden_states = self.exllama.forward(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    use_negative_cache=use_negative_cache
                )                
            else:
                logits, hidden_states = self.exllama.forward(input_ids=input_ids)
                
            if logits_need_squeezing:
                logits = logits[:, -1:, :]     
            
            return VibeVoiceCausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                last_hidden_state=hidden_states.to(self.dtype),
                attentions=None,
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.model.get_input_embeddings()(input_ids)
            
            if speech_tensors is not None and speech_masks is not None:
                acoustic_features, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
                if speech_input_mask is not None:
                    inputs_embeds[speech_input_mask] = speech_embeds         

            inputs_embeds = inputs_embeds.to(cache_position.device).to(self.dtype) 
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )  

            hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
            
            return VibeVoiceCausalLMOutputWithPast(
                logits=logits,
                past_key_values=outputs.past_key_values,
                last_hidden_state=hidden_states,
                attentions=outputs.attentions,
            )

    def _build_generate_config_model_kwargs(self, generation_config, inputs, tokenizer, return_processors=False, **kwargs):
        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )
        else:
            generation_config = GenerationConfig(
                **generation_config,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, True, 
            speech_start_id=tokenizer.speech_start_id, 
            speech_end_id=tokenizer.speech_end_id, 
            speech_diffusion_id=tokenizer.speech_diffusion_id, 
            **kwargs
        )
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id        

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        inputs_tensor = inputs_tensor.to(self.device)
        device = self.device
        
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)
        
        if return_processors:
            logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=None,
                logits_processor=LogitsProcessorList(),
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=StoppingCriteriaList())
        
            return generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria
        else:
            return generation_config, model_kwargs, input_ids
        
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        audio_streamer: Optional[Union[AudioStreamer, AsyncAudioStreamer]] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,        
        stop_check_fn: Optional[Callable[[], bool]] = None,        
        **kwargs,
    ) -> Union[torch.LongTensor, VibeVoiceGenerationOutput]:
        
        tokenizer = kwargs.pop("tokenizer", None)  
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        max_length_times = kwargs.pop("max_length_times", 2)

        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids, logits_processor, stopping_criteria = self._build_generate_config_model_kwargs(
            generation_config, inputs, tokenizer, return_processors=True, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, None, tokenizer, return_processors=False, **negative_kwargs
        )

        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None
        verbose = kwargs.get("verbose", False)

        audio_chunks = [[] for _ in range(batch_size)]
        initial_length = input_ids.shape[-1]
        initial_length_per_sample = model_kwargs['attention_mask'].sum(dim=-1)

        valid_tokens = [
            generation_config.speech_start_id,
            generation_config.speech_end_id, 
            generation_config.speech_diffusion_id,
            generation_config.eos_token_id
        ]
        if hasattr(generation_config, 'bos_token_id') and generation_config.bos_token_id is not None:
            valid_tokens.append(generation_config.bos_token_id)
        
        token_constraint_processor = VibeVoiceTokenConstraintProcessor(valid_tokens, device=device)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(token_constraint_processor)
        
        max_steps = min(generation_config.max_length - initial_length, int(max_length_times * initial_length))
        max_step_per_sample = torch.min(generation_config.max_length - initial_length_per_sample, (max_length_times * initial_length_per_sample).long())
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)

        use_exllama = self.exllama is not None        
        cpp_automaton = self._get_cpp_automaton() if use_exllama else None
        
        if kwargs.get("show_progress_bar", True):
            progress_bar = tqdm(range(max_steps), desc="Generating", leave=False)
        else:
            progress_bar = range(max_steps)
        
        inference_time_start = time.time()
        for step in progress_bar:            
            if stop_check_fn is not None:
                stop_check_fn()
            
            if audio_streamer is not None and hasattr(audio_streamer, 'finished_flags'):
                if any(audio_streamer.finished_flags):                    
                    if verbose: print(f"Audio generation stopped externally at step {step + 1}")
                    break
            
            if finished_tags.all():
                if hasattr(progress_bar, 'set_description'): progress_bar.set_description("Generation complete")
                break

            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopped it.")
                reached_samples = torch.arange(batch_size, device=device)[~finished_tags]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break            
            
            if hasattr(progress_bar, 'set_description'):
                active_samples = (~finished_tags).sum().item()
                progress_bar.set_description(f"Generating (active: {active_samples}/{batch_size})")

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device) if speech_tensors is not None else None,
                    "speech_masks": speech_masks.to(device) if speech_masks is not None else None,
                    "speech_input_mask": speech_input_mask.to(device) if speech_input_mask is not None else None,
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}

            past_len = 0 if step == 0 else input_ids.shape[-1] - 1
            
            outputs = self(
                **model_inputs,
                **prefill_inputs,
                logits_to_keep=1,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_exllama=(self.exllama is not None),
                past_len=past_len
            )            
            
            model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            
            if generation_config.do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            next_tokens[finished_tags] = generation_config.eos_token_id
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            if (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    finished_tags[new_eos_indices] = True
                    if verbose: print(f"Samples {new_eos_indices.tolist()} reached EOS token at step {step + 1}.", flush=True)
                    if audio_streamer is not None:
                        audio_streamer.end(new_eos_indices)

            max_length_reached = step >= max_step_per_sample
            new_max_length_indices = torch.nonzero(max_length_reached & ~finished_tags, as_tuple=False).squeeze(1)
            if new_max_length_indices.numel() > 0:
                finished_tags[new_max_length_indices] = True
                reach_max_step_sample[new_max_length_indices] = True
                if verbose: print(f"Samples {new_max_length_indices.tolist()} reached max generation length at step {step + 1}.", flush=True)
                if audio_streamer is not None:
                    audio_streamer.end(new_max_length_indices)

            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
            if diffusion_start_indices.numel() > 0 and kwargs.get('refresh_negative', True):
                for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
                    for sample_idx in diffusion_start_indices.tolist():
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id
            
            if use_exllama:   
                next_inputs_embeds = self.exllama.compute_inputs_embeds(next_tokens.unsqueeze(1))                
            else:
                next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)
            
            diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_diffusion_id)]            
            
            if diffusion_indices.numel() > 0:
                if kwargs.get('refresh_negative', True):
                    negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                    if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                        negative_model_inputs['inputs_embeds'] = inputs_embeds
                        negative_model_inputs['input_ids'] = None

                    past_len = self.exllama.negative_past_len if self.exllama is not None else 0
                    use_negative_cache = False                    
                    if step > 0 and self.negative_llm_steps_to_cache > 0:      
                        if step % self.negative_llm_steps_to_cache == 0:
                            use_negative_cache = False
                        elif self.negative_outputs_stored is not None:
                            use_negative_cache = True
                    
                    if use_negative_cache == False:  
                        self.negative_outputs_stored = self(
                            **negative_model_inputs, 
                            logits_to_keep=0, 
                            return_dict=True, 
                            output_attentions=False, 
                            output_hidden_states=False,
                            use_exllama=(self.exllama is not None),
                            use_negative_cache=True,  
                            past_len=past_len, 
                        )
                    negative_outputs = self.negative_outputs_stored  
                    negative_model_kwargs = self._update_model_kwargs_for_generation(negative_outputs, negative_model_kwargs, is_encoder_decoder=False)
                    negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                non_diffusion_mask = ~finished_tags & (next_tokens != generation_config.speech_diffusion_id)
                if non_diffusion_mask.any():
                    non_diffusion_indices = torch.arange(batch_size, device=device)[non_diffusion_mask]
                    start_indices = correct_cnt[non_diffusion_indices]

                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, negative_model_kwargs['past_key_values'].value_cache)):
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                    
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx+1:] = negative_input_ids[sample_idx, start_idx:-1].clone()
                                
                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]                
                
# ==== C++ EXLLAMAV3 EXTENSION ACCELERATION ====
                if use_exllama and cpp_automaton is not None:
                    
                    # --- DEEP TELEMETRY SETUP ---
                    if not hasattr(self, "_dbg_stats"):
                        self._dbg_stats = {"calls": 0, "last_time": time.perf_counter()}
                    
                    # Measure how long Python took to get here (LLM step + Overhead)
                    torch.cuda.synchronize() # Wait for LLM to finish before clocking
                    t_cpp_start = time.perf_counter()
                    py_overhead_duration = (t_cpp_start - self._dbg_stats["last_time"]) * 1000 # ms
                    
                    # --- 1st Knock: Diffusion & CFG ---
                    # Pointers only. No data transferred over PCIe.
                    speech_latent = cpp_automaton.sample(
                        positive_condition,
                        negative_condition,
                        float(cfg_scale),
                        True,
                        bool(self.increase_cfg)
                    ) # [B, 1, 64]
                    
                    # --- 2nd Knock: Acoustic Connector ---
                    acoustic_embed = cpp_automaton.acoustic_connector_forward(speech_latent.squeeze(1)).unsqueeze(1)
                    
                    # Sync to measure exact GPU execution time
                    torch.cuda.synchronize() 
                    t_cpp_end = time.perf_counter()
                    cpp_duration = (t_cpp_end - t_cpp_start) * 1000 # ms
                    
                    self._dbg_stats["calls"] += 1
                    
                    if verbose or self._dbg_stats["calls"] % 10 == 0:
                        pct_cpp = (cpp_duration / (cpp_duration + py_overhead_duration)) * 100 if (cpp_duration + py_overhead_duration) > 0 else 0
                        print(f"\n[Telemetry] Token {step+1} | "
                              f"Python LLM logic: {py_overhead_duration:.2f} ms | "
                              f"C++ Automaton (20 loops): {cpp_duration:.2f} ms ({pct_cpp:.1f}% of total) | "
                              f"PCIe Data: 0 bytes", flush=True)
                              
                    # Reset clock for next Python cycle
                    self._dbg_stats["last_time"] = time.perf_counter()

                else:
                    speech_latent = self.sample_speech_tokens(
                        positive_condition,
                        negative_condition,
                        cfg_scale=cfg_scale,
                        increase_cfg=self.increase_cfg,
                    ).unsqueeze(1) 
                                
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(speech_latent.device) - self.model.speech_bias_factor.to(speech_latent.device)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices.to(self.model.acoustic_tokenizer.device),
                    use_cache=True,
                    debug=False
                )                
                
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                if audio_streamer is not None:
                    audio_streamer.put(audio_chunk, diffusion_indices)
                    
                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean 
                
                # ==== C++ EXLLAMAV3 EXTENSION ACCELERATION ====
                if use_exllama and cpp_automaton is not None:
                    acoustic_embed = cpp_automaton.acoustic_connector_forward(speech_latent.squeeze(1)).unsqueeze(1)
                else:
                    acoustic_embed = self.model.acoustic_connector(speech_latent)
                    
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed
                diffusion_embeds = diffusion_embeds.to(next_inputs_embeds.dtype)

                next_inputs_embeds = next_inputs_embeds.to(diffusion_indices.device)
                next_inputs_embeds[diffusion_indices] = diffusion_embeds
            
            inputs_embeds = next_inputs_embeds

        if audio_streamer is not None:
            audio_streamer.end()

        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                final_audio_outputs.append(None)
        
        print(f"segment inference took: {time.time() - inference_time_start:.2f} s.")
        if use_exllama and self.exllama is not None:
            self.exllama.reset_all()
        
        return VibeVoiceGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )
    
    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=1.3, increase_cfg=False):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        batch_size = condition.shape[0]
        device = self.model.prediction_head.device
        dtype = self.model.prediction_head.dtype
        conditions = torch.cat([condition, neg_condition], dim=0).to(device=device, dtype=dtype)
        speech = torch.randn((batch_size, self.config.acoustic_vae_dim), device=device, dtype=dtype)
        total_steps = len(self.model.noise_scheduler.timesteps)
        
        for i, t in enumerate(self.model.noise_scheduler.timesteps):
            latent_model_input = torch.cat([speech] * 2).to(dtype)
            model_output = self.model.prediction_head(
                latent_model_input,
                t.repeat(latent_model_input.shape[0]).to(latent_model_input),
                condition=conditions
            )
            cond_pred, uncond_pred = model_output.chunk(2)
            if increase_cfg:
                progress = i / total_steps
                current_cfg_scale = cfg_scale * (1.0 + 0.5 * (progress < 0.5)) 
            else:
                current_cfg_scale = cfg_scale
            guided_pred = uncond_pred + current_cfg_scale * (cond_pred - uncond_pred)
            speech = self.model.noise_scheduler.step(guided_pred, t, speech).prev_sample
        return speech    

AutoModelForCausalLM.register(VibeVoiceConfig, VibeVoiceForConditionalGenerationInference)

__all__ = [
    "VibeVoiceForConditionalGenerationInference", "ExLlamaV3Wrapper"
]