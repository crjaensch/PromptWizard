#!/usr/bin/env python3
"""
Self-generated Chain of Thought (CoT) Steps

This module provides functionality to generate Chain of Thought reasoning steps
for examples, optimizing both the instruction prompts and the examples themselves.
It is extracted from the PromptWizard framework and made self-contained.
"""

from typing import List, Dict, Tuple, Optional
from pathlib import Path
import sys

# Add the directory containing this file to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared utilities
from prompt_opt_utils import (
    LLMManager, DataProcessor, BasePromptGenerator,
    ANSWER_START, ANSWER_END, QUESTION_LITERAL, 
    ANSWER_WITH_REASON_LITERAL, FINAL_ANSWER_LITERAL
)

# Prompt templates
PROMPT_TEMPLATES = {
    "system_prompt": "You are a helpful assistant developed by OpenAI that can efficiently perform tasks as per instruction",
    
    "expert_profile": "You are a helpful assistant developed by OpenAI that can efficiently perform tasks as per instruction",
    
    "expert_template": """For each instruction, write a high-quality description about the most capable and suitable agent to answer the instruction. In second person perspective.

[Instruction]: {task_description}
[Agent Description]:""",
    
    "generate_reason_template": """You are given a task description and instructions of given task

[Task Description]: {task_description}

[Instruction]: {instruction}

Each example has a question denoted by a question [Question] and a final answer [Answer].

[Question]: {question}

[Answer]: {answer}

Please explain your reasoning behind reaching the answer given in a concise, complete, and coherent text of reasoning that contains all the steps or logical pathways followed. Ensure it is specific and non-ambiguous, and assume the necessary domain knowledge is in the question and task description.

[Improved Reasoning Chain]:""",
    
    "reason_optimization_template": """You are given a task description and instructions of given task

[Task Description]: {task_description}

[Instruction]: {instruction}

Each example has a question denoted by a question [Question] and a final answer [Answer].

[Question]: {question}

[Answer]: {answer}

Please explain your reasoning behind reaching the answer given in a concise, complete, and coherent text of reasoning that contains all the steps or logical pathways followed. Ensure it is specific and non-ambiguous, and assume the necessary domain knowledge is in the question and task description.

[Improved Reasoning Chain]:""",
    
    "intent_template": """You are given an instruction along description of task labelled as [Task Description]. For the given instruction, list out 3-5 keywords in comma separated format as [Intent] which define the characteristics or properties required by the about the most capable and suitable agent to solve the task using the instruction.

[Task Description]: {task_description}
[Instruction]: {instruction}

[Intent]:""",
    
    "quest_reason_ans": """[Question] {question}
[Answer] {answer}""",
    
    "final_prompt": """{instruction}
{few_shot_examples}

{answer_format}"""
}


class ChainOfThoughtGenerator(BasePromptGenerator):
    """Generates Chain of Thought reasoning for examples."""
    
    def __init__(self, llm_manager: LLMManager, data_processor: DataProcessor):
        """
        Initialize the Chain of Thought Generator.
        
        Args:
            llm_manager: Instance of LLMManager for API calls
            data_processor: Instance of DataProcessor for data handling
        """
        super().__init__(llm_manager, data_processor)
        self.templates = PROMPT_TEMPLATES
    
    def generate_reasoning(self, task_description: str, instruction: str, question: str, answer: str) -> str:
        """
        Generate reasoning for a given question and answer.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            question: The question to generate reasoning for
            answer: The answer to explain
            
        Returns:
            Generated reasoning text
        """
        prompt_template = self.templates["generate_reason_template"].format(
            task_description=task_description,
            instruction=instruction,
            question=question,
            answer=answer
        )
        return self.chat_completion(user_prompt=prompt_template)
    
    def generate_expert_identity(self, task_description: str) -> str:
        """
        Generate an expert identity description for the given task.
        
        Args:
            task_description: Description of the task
            
        Returns:
            Expert identity description
        """
        return super().generate_expert_identity(task_description, self.templates["expert_template"])
    
    def generate_intent_keywords(self, task_description: str, instruction: str) -> str:
        """
        Generate intent keywords for the given task and instruction.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            
        Returns:
            Intent keywords string
        """
        return super().generate_intent_keywords(task_description, instruction, self.templates["intent_template"])
    
    def add_reasoning_to_examples(self, examples: List[Dict], task_description: str, instruction: str) -> List[Dict]:
        """
        Add reasoning to a list of examples.
        
        Args:
            examples: List of example dictionaries
            task_description: Description of the task
            instruction: Instruction for solving the task
            
        Returns:
            Examples with added reasoning
        """
        examples_with_reasoning = []
        for example in examples:
            reason = self.generate_reasoning(
                task_description,
                instruction,
                example[QUESTION_LITERAL],
                example[FINAL_ANSWER_LITERAL]
            )
            
            example_with_reasoning = example.copy()
            example_with_reasoning[ANSWER_WITH_REASON_LITERAL] = f"{reason} {ANSWER_START}{example[FINAL_ANSWER_LITERAL]}{ANSWER_END}"
            examples_with_reasoning.append(example_with_reasoning)
            
        return examples_with_reasoning
    
    def create_prompt_with_cot_examples(self, task_description: str, instruction: str, 
                                       examples: List[Dict], answer_format: str) -> Tuple[str, str]:
        """
        Create a prompt with Chain of Thought examples.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            examples: List of example dictionaries
            answer_format: Format specification for answers
            
        Returns:
            Tuple of (final_prompt, expert_identity)
        """
        # Generate expert identity
        expert_identity = self.generate_expert_identity(task_description)
        
        # Add reasoning to examples
        examples_with_reasoning = self.add_reasoning_to_examples(examples, task_description, instruction)
        
        # Format examples as string
        few_shot_examples = self.data_processor.collate_to_str(
            examples_with_reasoning, 
            self.templates["quest_reason_ans"]
        )
        
        # Generate intent keywords
        intent_keywords = self.generate_intent_keywords(task_description, instruction)
        
        # Create final prompt
        final_prompt = self.templates["final_prompt"].format(
            instruction=instruction,
            few_shot_examples=few_shot_examples,
            answer_format=answer_format
        )
        
        final_prompt += "\nKeywords: " + intent_keywords
        
        return final_prompt, expert_identity


class SelfGeneratedCoT:
    """Main class for Self-generated Chain of Thought implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4"):
        """
        Initialize the Self-generated CoT module.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
            model_name: Name of the model to use.
        """
        self.llm_manager = LLMManager(api_key, model_name)
        self.data_processor = DataProcessor()
        self.cot_generator = ChainOfThoughtGenerator(self.llm_manager, self.data_processor)
    
    def optimize_prompt_with_cot(self, task_description: str, instruction: str, 
                               examples: List[Dict], answer_format: str) -> Tuple[str, str]:
        """
        Optimize a prompt with Chain of Thought reasoning.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            examples: List of example dictionaries with 'question' and 'final_answer' keys
            answer_format: Format specification for answers
            
        Returns:
            Tuple of (optimized_prompt, expert_identity)
        """
        return self.cot_generator.create_prompt_with_cot_examples(
            task_description, instruction, examples, answer_format
        )
    
    def generate_synthetic_examples_with_cot(self, task_description: str, instruction: str, 
                                          num_examples: int = 3) -> List[Dict]:
        """
        Generate synthetic examples with Chain of Thought reasoning.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            num_examples: Number of examples to generate
            
        Returns:
            List of synthetic examples with reasoning
        """
        # This is a simplified implementation - in a real scenario, you would use a more
        # sophisticated approach to generate diverse and high-quality examples
        prompt = f"""You are an expert at generating examples for the following task:
        
        [Task Description]: {task_description}
        [Instruction]: {instruction}
        
        Please generate {num_examples} diverse and high-quality examples for this task.
        For each example, provide a question and its answer.
        
        Format each example as follows:
        <START>
        [Question] Your question here
        [Answer] Your detailed reasoning here {ANSWER_START}Your final answer here{ANSWER_END}
        <END>
        """
        
        response = self.cot_generator.chat_completion(prompt)
        examples = self.data_processor.extract_examples_from_response(response)
        
        return examples
    
    def solve_with_cot_prompt(self, prompt: str, expert_identity: str, question: str) -> str:
        """
        Solve a question using a Chain of Thought optimized prompt.
        
        Args:
            prompt: The optimized prompt
            expert_identity: Expert identity description
            question: Question to solve
            
        Returns:
            LLM response with reasoning and answer
        """
        full_prompt = f"{prompt}\n\n[Question] {question}\n[Answer] "
        return self.cot_generator.chat_completion(full_prompt, expert_identity)


# Example usage
def example_usage():
    """Example usage of the Self-generated CoT module."""
    # Initialize the module
    cot = SelfGeneratedCoT()
    
    # Define task and instruction
    task_description = "You are a mathematics expert. You will be given a mathematics problem which you need to solve"
    instruction = "Let's think step by step to solve this math problem."
    answer_format = "At the end, wrap only your final answer between <ANS_START> and <ANS_END> tags"
    
    # Define some example questions and answers
    examples = [
        {
            "question": "If 5x + 3 = 18, what is the value of x?",
            "final_answer": "3"
        },
        {
            "question": "What is the area of a circle with radius 4 cm?",
            "final_answer": "16 * PI square cm"
        }
    ]
    
    # Optimize the prompt with CoT
    optimized_prompt, expert_identity = cot.optimize_prompt_with_cot(
        task_description, instruction, examples, answer_format
    )
    
    print("Optimized Prompt:")
    print(optimized_prompt)
    print("\nExpert Identity:")
    print(expert_identity)
    
    # Generate synthetic examples
    synthetic_examples = cot.generate_synthetic_examples_with_cot(
        task_description, instruction
    )
    
    print("\nSynthetic Examples:")
    for i, example in enumerate(synthetic_examples):
        print(f"Example {i+1}:")
        print(f"Question: {example['question']}")
        print(f"Answer with Reasoning: {example['answer']}")
        print(f"Final Answer: {example['final_answer']}")
        print()
    
    # Solve a new question
    new_question = "If the sum of three consecutive integers is 42, what is the middle integer?"
    solution = cot.solve_with_cot_prompt(optimized_prompt, expert_identity, new_question)
    
    print("Solution to New Question:")
    print(solution)


if __name__ == "__main__":
    # Uncomment to run the example
    example_usage()
    # pass
