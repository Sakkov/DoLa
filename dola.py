import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
import numpy as np

class LLamaQaStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for LLaMA."""
    def __init__(self, list_stop_word_ids):
        super().__init__()
        self.list_stop_word_ids = list_stop_word_ids
        
    def __call__(self, input_ids, scores, **kwargs):
        if not self.list_stop_word_ids:
            return False
            
        for stop_ids in self.list_stop_word_ids:
            if len(stop_ids) <= input_ids.shape[1]:
                # Compare the last len(stop_ids) tokens with stop_ids
                if torch.all((input_ids[0, -len(stop_ids):] == torch.tensor(stop_ids).to(input_ids.device))):
                    return True
        return False

class DoLa:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        """Load the model and tokenizer."""
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True, 
            **kwargs
        )

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer

    def set_stop_words(self, stop_words):
        """Set stop words for generation."""
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word, add_special_tokens=False)
            list_stop_word_ids.append(stop_word_ids)
            print(f"Added stop word: {stop_word} with the ids {stop_word_ids}", flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def map_mode_to_dola_layers(self, mode, mature_layer=None, premature_layer=None, candidate_premature_layers=None):
        """Map the original DoLa modes to the new dola_layers parameter."""
        # First, determine the number of layers in the model
        config = self.model.config
        num_layers = getattr(config, "num_hidden_layers", None)
        if num_layers is None:
            # Try alternative attribute names based on model architecture
            num_layers = getattr(config, "n_layers", 
                      getattr(config, "num_layers", 
                      getattr(config, "n_layer", 32)))  # Default to 32 if not found
        
        print(f"Model has {num_layers} layers")
        
        # Adjust layer indices to be within bounds
        if mature_layer is not None and mature_layer >= num_layers:
            mature_layer = num_layers - 1
            print(f"Adjusting mature_layer to {mature_layer} (model's last layer)")
            
        if premature_layer is not None and premature_layer >= num_layers:
            premature_layer = num_layers - 2  # Default to second-to-last layer
            print(f"Adjusting premature_layer to {premature_layer}")
            
        if candidate_premature_layers:
            adjusted_candidates = []
            for layer in candidate_premature_layers:
                if layer < num_layers:
                    adjusted_candidates.append(layer)
                else:
                    print(f"Skipping candidate layer {layer} (out of bounds)")
            candidate_premature_layers = adjusted_candidates
            if not candidate_premature_layers:
                # If all candidates were out of bounds, use a default set
                candidate_premature_layers = [i for i in range(0, num_layers-1, max(1, num_layers//8))]
                print(f"Using default candidate layers: {candidate_premature_layers}")
        
        # Map the mode to appropriate parameters
        if mode == "baseline":
            return None  # No DoLa decoding
        elif mode == "dola-static":
            # Using the specific mature and premature layers
            if mature_layer is None or premature_layer is None:
                raise ValueError("For dola-static mode, both mature_layer and premature_layer must be specified")
            return {"mature_layer": mature_layer, "premature_layer": premature_layer}
        elif mode == "dola":
            # Dynamic selection of premature layers
            if mature_layer is None or not candidate_premature_layers:
                raise ValueError("For dola mode, mature_layer and candidate_premature_layers must be specified")
            return {"mature_layer": mature_layer, "candidate_layers": candidate_premature_layers}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate(self, input_text, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, 
                 mature_layer=None, premature_layer=None, candidate_premature_layers=None, 
                 mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        """Generate text using DoLa decoding."""
        with torch.no_grad():
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens
            
            # Map the DoLa mode to the appropriate parameters
            dola_config = self.map_mode_to_dola_layers(mode, mature_layer, premature_layer, candidate_premature_layers)
            
            generation_kwargs = {
                "max_length": max_len,
                "num_return_sequences": 1,
                "output_scores": True,
                "return_dict_in_generate": True,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "stopping_criteria": self.stopping_criteria,
                **kwargs
            }
            
            # Apply DoLa configuration if provided
            if dola_config is not None:
                if mode == "dola-static":
                    # For static mode, specify the mature and premature layers
                    generation_kwargs["dola_mature_layer"] = dola_config["mature_layer"]
                    generation_kwargs["dola_premature_layer"] = dola_config["premature_layer"]
                    generation_kwargs["dola_relative_top"] = relative_top
                elif mode == "dola":
                    # For dynamic mode, specify the mature layer and candidate layers
                    generation_kwargs["dola_mature_layer"] = dola_config["mature_layer"]
                    generation_kwargs["dola_candidate_layers"] = dola_config["candidate_layers"]
                    generation_kwargs["dola_relative_top"] = relative_top
            
            outputs = self.model.generate(input_ids, **generation_kwargs)
            
            sequences, scores = outputs.sequences, outputs.scores
            
            # Extract the generated text
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)
            
            if verbose:
                print(f'MODEL OUTPUT: \n{output_str}')
                
            # Remove stop words if requested
            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()
                
            # Get the distribution of premature layers if available
            premature_layer_dist = getattr(outputs, 'premature_layer_dist', None)

        if self.device:
            torch.cuda.empty_cache()
            
        return output_str, premature_layer_dist

    def get_relative_top_filter(self, scores, relative_top=0.1, min_tokens_to_keep=1):
        """Create a filter for tokens based on their relative probability."""
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, 
                 temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=None, 
                 mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, 
                 relative_top_value=-1000.0, post_softmax=True, **kwargs):
        """Calculate language model scores using DoLa."""
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            
            # Default case: use the final layer
            if mode == 'baseline':
                outputs = self.model(input_ids)
                logits = outputs.logits.squeeze(0)
                logits = logits.log_softmax(dim=-1)
                
                # Skip tokens in the prompt -- we only care about the answer
                logits = logits[prefix_ids.shape[-1] - 1: -1, :]
                
                # Get logprobs for each token in the answer
                log_probs = logits[range(logits.shape[0]), continue_ids].sum().item()
                return log_probs, None
                
            # For DoLa modes, we need to get hidden states from different layers
            else:
                # Add output_hidden_states=True for accessing all layers
                outputs = self.model(input_ids, output_hidden_states=True)
                
                if mode == 'dola-static':
                    # Get logits from the mature and premature layers
                    hidden_states = outputs.hidden_states
                    
                    # Get the total number of hidden states and adjust indices accordingly
                    num_hidden_states = len(hidden_states)
                    
                    # In some transformer models, hidden_states might be structured differently
                    # Let's make sure we don't go out of bounds
                    mature_idx = min(mature_layer, num_hidden_states - 1)
                    premature_idx = min(premature_layer, num_hidden_states - 1)
                    
                    # Get the base and final logits
                    base_hidden = hidden_states[premature_idx]
                    final_hidden = hidden_states[mature_idx]
                    
                    # Project hidden states to vocabulary space
                    # This assumes the model has a lm_head attribute, which is common in HuggingFace models
                    lm_head = self.model.get_output_embeddings()
                    base_logits = lm_head(base_hidden).squeeze(0)[prefix_ids.shape[-1] - 1: -1, :]
                    final_logits = lm_head(final_hidden).squeeze(0)[prefix_ids.shape[-1] - 1: -1, :]
                    
                    # Apply log_softmax
                    base_logits = base_logits.log_softmax(dim=-1)
                    final_logits = final_logits.log_softmax(dim=-1)
                    
                    # Calculate DoLa scores
                    diff_logits = final_logits - base_logits
                    if post_softmax:
                        diff_logits = diff_logits.log_softmax(dim=-1)
                    
                    # Apply relative top filter
                    if relative_top > 0.0:
                        relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                        diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                    # Calculate log probabilities for the continuation
                    log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                    return log_probs, None
                    
                elif mode == 'dola':
                    # Initialize premature layer distribution
                    premature_layer_dist = {l: 0 for l in candidate_premature_layers}
                    hidden_states = outputs.hidden_states
                    
                    # Get the total number of hidden states and adjust indices accordingly
                    num_hidden_states = len(hidden_states)
                    
                    # In some transformer models, hidden_states might be structured differently
                    # Let's make sure we don't go out of bounds
                    mature_idx = min(mature_layer, num_hidden_states - 1)
                    candidate_indices = [min(l, num_hidden_states - 1) for l in candidate_premature_layers]
                    
                    # Get the lm_head for projecting hidden states to vocabulary
                    lm_head = self.model.get_output_embeddings()
                    
                    # For each token in the continuation
                    final_logits = []
                    base_logits = []
                    
                    for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                        # Get mature layer representation for this token
                        mature_hidden = hidden_states[mature_idx][:, seq_i, :]
                        mature_logits = lm_head(mature_hidden)
                        
                        # Stack all premature layers for this token
                        premature_hiddens = torch.stack([hidden_states[i][:, seq_i, :] for i in candidate_indices], dim=0)
                        premature_logits = lm_head(premature_hiddens)
                        
                        # Calculate softmax values
                        softmax_mature = F.softmax(mature_logits, dim=-1)
                        softmax_premature = F.softmax(premature_logits, dim=-1)
                        
                        # Calculate M, the average distribution
                        M = 0.5 * (softmax_mature.unsqueeze(0) + softmax_premature)
                        
                        # Calculate log-softmax for the KL divergence
                        log_softmax_mature = F.log_softmax(mature_logits, dim=-1)
                        log_softmax_premature = F.log_softmax(premature_logits, dim=-1)
                        
                        # Calculate the KL divergences and then the JS divergences
                        kl1 = F.kl_div(log_softmax_mature.unsqueeze(0), M, reduction='none').mean(-1)
                        kl2 = F.kl_div(log_softmax_premature, M, reduction='none').mean(-1)
                        js_divs = 0.5 * (kl1 + kl2)
                        
                        # Find the premature layer with max JS divergence
                        max_js_idx = js_divs.argmax().item()
                        selected_premature_layer = candidate_premature_layers[max_js_idx]
                        premature_layer_dist[selected_premature_layer] += 1
                        
                        # Store the selected logits
                        final_logits.append(mature_logits)
                        base_logits.append(premature_logits[max_js_idx])
                    
                    # Convert to tensors and compute difference
                    final_logits = torch.stack(final_logits).squeeze(1)
                    base_logits = torch.stack(base_logits).squeeze(1)
                    
                    # Apply log_softmax
                    final_logits = final_logits.log_softmax(dim=-1)
                    base_logits = base_logits.log_softmax(dim=-1)
                    
                    # Calculate DoLa scores
                    diff_logits = final_logits - base_logits
                    if post_softmax:
                        diff_logits = diff_logits.log_softmax(dim=-1)
                    
                    # Apply relative top filter
                    if relative_top > 0.0:
                        relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                        diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                    # Calculate log probabilities for the continuation
                    log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                    return log_probs, premature_layer_dist