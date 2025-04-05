import yaml
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Import the necessary classes
from critique_refine_with_samples import CritiqueNRefinePromptPool, PromptPool


def load_prompt_pool_from_yaml(yaml_path: str) -> CritiqueNRefinePromptPool:
    """
    Load a CritiqueNRefinePromptPool configuration from a YAML file.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        A configured CritiqueNRefinePromptPool instance
    """
    # Load the YAML file
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract thinking styles as a list
    thinking_styles = config.get('thinking_styles', [])
    
    # Create the prompt pool instance
    prompt_pool = CritiqueNRefinePromptPool(
        # Base prompt templates
        final_prompt=config['base_templates'].get('final_prompt', ''),
        eval_prompt=config['base_templates'].get('eval_prompt', ''),
        system_prompt=config['base_templates'].get('system_prompt', ''),
        
        # Core templates
        quest_reason_ans=config['core_templates'].get('quest_reason_ans', ''),
        expert_profile=config['core_templates'].get('expert_profile', ''),
        ans_delimiter_instruction=config['core_templates'].get('ans_delimiter_instruction', ''),
        
        # Thinking styles
        thinking_styles=thinking_styles,
        
        # Critique templates
        meta_critique_template=config['critique_templates'].get('meta_critique_template', ''),
        meta_positive_critique_template=config['critique_templates'].get('meta_positive_critique_template', ''),
        critique_refine_template=config['critique_templates'].get('critique_refine_template', ''),
        
        # Generation templates
        solve_template=config['generation_templates'].get('solve_template', ''),
        meta_sample_template=config['generation_templates'].get('meta_sample_template', ''),
        intent_template=config['generation_templates'].get('intent_template', ''),
        expert_template=config['generation_templates'].get('expert_template', ''),
        
        # Reasoning templates
        generate_reason_template=config['reasoning_templates'].get('generate_reason_template', ''),
        reason_optimization_template=config['reasoning_templates'].get('reason_optimization_template', ''),
        
        # Example templates
        examples_optimization_template=config['example_templates'].get('examples_optimization_template', ''),
        examples_critique_template=config['example_templates'].get('examples_critique_template', ''),
        examples_critique_template_zero_shot=config['example_templates'].get('examples_critique_template_zero_shot', '')
    )
    
    return prompt_pool


def save_prompt_pool_to_yaml(prompt_pool: CritiqueNRefinePromptPool, yaml_path: str) -> None:
    """
    Save a CritiqueNRefinePromptPool configuration to a YAML file.
    
    Args:
        prompt_pool: The CritiqueNRefinePromptPool instance to save
        yaml_path: Path where the YAML configuration will be saved
    """
    # Create the configuration dictionary
    config = {
        'base_templates': {
            'final_prompt': prompt_pool.final_prompt,
            'eval_prompt': prompt_pool.eval_prompt,
            'system_prompt': prompt_pool.system_prompt,
        },
        'core_templates': {
            'quest_reason_ans': prompt_pool.quest_reason_ans,
            'expert_profile': prompt_pool.expert_profile,
            'ans_delimiter_instruction': prompt_pool.ans_delimiter_instruction,
        },
        'thinking_styles': prompt_pool.thinking_styles,
        'critique_templates': {
            'meta_critique_template': prompt_pool.meta_critique_template,
            'meta_positive_critique_template': prompt_pool.meta_positive_critique_template,
            'critique_refine_template': prompt_pool.critique_refine_template,
        },
        'generation_templates': {
            'solve_template': prompt_pool.solve_template,
            'meta_sample_template': prompt_pool.meta_sample_template,
            'intent_template': prompt_pool.intent_template,
            'expert_template': prompt_pool.expert_template,
        },
        'reasoning_templates': {
            'generate_reason_template': prompt_pool.generate_reason_template,
            'reason_optimization_template': prompt_pool.reason_optimization_template,
        },
        'example_templates': {
            'examples_optimization_template': prompt_pool.examples_optimization_template,
            'examples_critique_template': prompt_pool.examples_critique_template,
            'examples_critique_template_zero_shot': prompt_pool.examples_critique_template_zero_shot,
        },
    }
    
    # Save to YAML file
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# Example usage
if __name__ == "__main__":
    # Load a prompt pool from YAML
    yaml_path = "critique-refine-promptpool.yaml"
    prompt_pool = load_prompt_pool_from_yaml(yaml_path)
    
    # Use the prompt pool...
    
    # Save a modified prompt pool back to YAML
    save_prompt_pool_to_yaml(prompt_pool, "modified-prompt-pool.yaml")
