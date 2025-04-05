#!/usr/bin/env python3

"""Example of using the Critique and Refine methodology with YAML configuration"""

import os
from typing import List

# Import the necessary components
from critique_refine_with_samples import CritiqueNRefine, extract_between, DatasetSpecificProcessing
from critique_refine_promptpool_loader import load_prompt_pool_from_yaml
from dataclasses import dataclass

@dataclass
class PromptOptimizationParams:
    # Number of candidate prompts to generate in given iteration
    style_variation: int
    # Number of questions to be asked to LLM in a single go
    questions_batch_size: int
    # Number of batches of questions to correctly answered, for a prompt to be considered as performing good
    min_correct_count: int
    # Max number of mini-batches on which we should evaluate our prompt
    max_eval_batches: int
    # Number of top best performing prompts to be considered for next iterations
    top_n: int
    # Number of rounds of mutation to be performed when generating different styles
    mutation_rounds: int
    # Refine instruction post mutation
    refine_instruction: bool
    # Number of iterations for conducting <mutation_rounds> rounds of mutation of task description
    # followed by refinement of instructions
    mutate_refine_iterations: int
    # Number of iterations for refining task description and in context examples for few-shot
    refine_task_eg_iterations: int
    # Description of task. This will be fed to prompt
    task_description: str
    # Base instruction, in line with your dataset. This will be fed to prompt
    base_instruction: str
    # Instruction for specifying answer format
    answer_format: str
    # Number of samples from dataset, set aside as training data. In every iteration we would be drawing
    # `questions_batch_size` examples from training data with replacement.
    seen_set_size: int
    # Number of examples to be given for few shots
    few_shot_count: int
    # Generate synthetic reasoning
    generate_reasoning: bool
    # Generate description of an expert which can solve the task at hand
    generate_expert_identity: bool
    # Generate keywords that describe the intent of the task
    generate_intent_keywords: bool
    # number of synthetic training examples to be generated
    num_train_examples: int


def main():
    """Main function to demonstrate the Critique and Refine process with YAML configuration"""
    # Load the prompt pool from YAML configuration
    yaml_path = os.path.join(os.path.dirname(__file__), "critique-refine-promptpool.yaml")
    critique_prompt_pool = load_prompt_pool_from_yaml(yaml_path)
    
    # Create a dataset-specific processor
    class MyDataProcessor(DatasetSpecificProcessing):
        def extract_final_answer(self, answer: str) -> str:
            # Implement your custom logic to extract the final answer
            return extract_between(answer, "<ANS_START>", "<ANS_END>")
    
    # Create your dataset
    my_dataset = [
        {
            "question": "What is 2+2?",
            "answer": "To solve this addition problem, I need to add 2 and 2 together. <ANS_START>4<ANS_END>",
            "final_answer": "4"
        },
        {
            "question": "What is 5+7?",
            "answer": "I need to add 5 and 7. 5+7=12. <ANS_START>12<ANS_END>",
            "final_answer": "12"
        },
        {
            "question": "What is 10-3?",
            "answer": "I need to subtract 3 from 10. 10-3=7. <ANS_START>7<ANS_END>",
            "final_answer": "7"
        },
        # Add more examples as needed
    ]
    
    # Create optimization parameters
    optimization_params = PromptOptimizationParams(
        style_variation=5,
        questions_batch_size=3,
        min_correct_count=2,
        max_eval_batches=5,
        top_n=3,
        mutation_rounds=2,
        refine_instruction=True,
        mutate_refine_iterations=3,
        refine_task_eg_iterations=2,
        task_description="Solve basic arithmetic problems",
        base_instruction="Calculate the result of the given arithmetic expression.",
        answer_format="Provide your answer between <ANS_START> and <ANS_END> tags.",
        seen_set_size=3,  # Using a small number since we have few examples
        few_shot_count=1,
        generate_reasoning=True,
        generate_expert_identity=True,
        generate_intent_keywords=True,
        num_train_examples=2
    )
    
    # Initialize the CritiqueNRefine class
    data_processor = MyDataProcessor()
    critique_refine = CritiqueNRefine(
        dataset=my_dataset,
        prompt_pool=critique_prompt_pool,
        data_processor=data_processor
    )
    
    # Get the best prompt
    best_prompt, best_score = critique_refine.get_best_prompt(
        params=optimization_params,
        use_examples=True
    )
    
    print(f"Best prompt (score: {best_score}):\n{best_prompt}")


if __name__ == "__main__":
    main()