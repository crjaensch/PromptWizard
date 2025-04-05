#!/usr/bin/env python3
"""
Prompt Optimization Utilities

This module provides common utilities for prompt optimization techniques used in PromptWizard.
It contains shared functionality between the Critique and Refine and Self-generated CoT methods.
"""

import re
import os
from typing import List, Dict, Any, Tuple, Optional, Union

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ===============================================================================
# Constants
# ===============================================================================

# Text processing constants
TEXT_DELIMITER_PATTERN = r"(?s)(?<=<START>)(.*?)(?=<END>)"
ANSWER_START = "<ANS_START>"
ANSWER_END = "<ANS_END>"
ANSWER_DELIMITER_PATTERN = r"(?s)(?<="+ANSWER_START+")(.*?)(?="+ANSWER_END+")"
QUESTION_LITERAL = "question"
ANSWER_WITH_REASON_LITERAL = "answer"
FINAL_ANSWER_LITERAL = "final_answer"
QUESTION_KEY_IN_PROMPT = "[Question]"
ANSWER_KEY_IN_PROMPT = "[Answer]"

# ===============================================================================
# Utility Functions
# ===============================================================================

def extract_between(start: str, end: str, text: str) -> str:
    """
    Extracts the substring from 'text' that is between 'start' and 'end' strings.
    
    Parameters:
    - start (str): The starting delimiter string.
    - end (str): The ending delimiter string.
    - text (str): The text to search within.
    
    Returns:
    - str: The extracted substring between the start and end delimiters.
    """
    start_index = text.find(start)
    if start_index == -1:
        return '' 
    
    start_index += len(start)
    
    end_index = text.find(end, start_index)
    if end_index == -1:
        return ''  
    return text[start_index:end_index]


# ===============================================================================
# LLM Manager
# ===============================================================================

class LLMManager:
    """Manages interactions with language models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4"):
        """
        Initialize the LLM Manager.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
            model_name: Name of the model to use.
        """
        self.model_name = model_name
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'.")
        
        if api_key is None:
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set the OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Make a chat completion request to the OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            
        Returns:
            The content of the assistant's response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return ""


# ===============================================================================
# Data Processing
# ===============================================================================

class DataProcessor:
    """Processes data for prompt optimization techniques."""
    
    def normalize_prediction(self, prediction: str, lowercase: bool = True) -> str:
        """
        Normalize a prediction string for comparison.
        
        Args:
            prediction: The prediction string to normalize.
            lowercase: Whether to convert to lowercase.
            
        Returns:
            Normalized prediction string.
        """
        import string
        prediction = prediction.replace(' and ', ' ')
        prediction = prediction.replace('Sentence 1:', ' ')
        prediction = prediction.replace('Sentence 2:', ' ')
        prediction = prediction.strip()
        prediction = prediction.split("\n")[0]
        prediction = prediction.split(".")[0]

        if lowercase:
            prediction = prediction.lower()

        # remove punctuation
        prediction = prediction.replace('-', ' ')
        prediction = prediction.translate(
            str.maketrans('', '', string.punctuation))

        return prediction
    
    def access_answer(self, llm_output: str, gt_answer: str) -> Tuple[bool, Any]:
        """
        Compare answer generated by model with the answer in ground truth.
        
        Args:
            llm_output: Output of LLM (the predicted answer)
            gt_answer: The expected ground truth answer
            
        Returns:
            Tuple of (is_correct, predicted_answer)
        """
        predicted_answer = self.extract_final_answer(llm_output)
        is_correct = False
        if predicted_answer and (predicted_answer.lower() == gt_answer.lower()):
            is_correct = True

        return is_correct, predicted_answer
    
    def collate_to_str(self, examples: List[Dict], example_template: str) -> str:
        """
        Collate examples into a string using the provided template.
        
        Args:
            examples: List of example dictionaries
            example_template: Template string for formatting examples
            
        Returns:
            Collated string of examples
        """
        example_string = ""
        for example in examples:
            answer = example[FINAL_ANSWER_LITERAL]
            if ANSWER_WITH_REASON_LITERAL in example:
                answer = example[ANSWER_WITH_REASON_LITERAL]

            example_string += example_template.format(
                question=example[QUESTION_LITERAL],
                answer=answer
            )
        return example_string
    
    def extract_final_answer(self, answer: str) -> str:
        """
        Extract the final answer from the LLM output.
        
        Args:
            answer: The LLM output string
            
        Returns:
            Extracted final answer
        """
        # Look for answer between delimiters
        extracted = extract_between(ANSWER_START, ANSWER_END, answer)
        if extracted:
            return extracted.strip()
        
        # If no delimiters, return the whole answer
        return answer.strip()
    
    def extract_examples_from_response(self, response_with_examples: str) -> List[Dict]:
        """
        Extract examples from LLM response.
        
        Args:
            response_with_examples: LLM response containing examples
            
        Returns:
            List of extracted example dictionaries
        """
        synthetic_examples = []
        parsed_data = re.findall(TEXT_DELIMITER_PATTERN, response_with_examples, re.DOTALL)
        parsed_data = [s.strip() for s in parsed_data]

        for text in parsed_data:
            # Splitting text into question, reason, and answer
            if QUESTION_KEY_IN_PROMPT in text and ANSWER_KEY_IN_PROMPT in text:
                question = text[text.find(QUESTION_KEY_IN_PROMPT) +
                                len(QUESTION_KEY_IN_PROMPT):
                                text.find(ANSWER_KEY_IN_PROMPT)].strip()
                answer_with_reason = text[text.find(ANSWER_KEY_IN_PROMPT) +
                                          len(ANSWER_KEY_IN_PROMPT):].strip()

                final_answer = extract_between(text=answer_with_reason, start=ANSWER_START, end=ANSWER_END)
                if not final_answer:
                    final_answer = answer_with_reason

                formatted_data = {
                    QUESTION_LITERAL: question,
                    ANSWER_WITH_REASON_LITERAL: answer_with_reason,
                    FINAL_ANSWER_LITERAL: final_answer
                }

                synthetic_examples.append(formatted_data)

        return synthetic_examples


# ===============================================================================
# Base Prompt Generator
# ===============================================================================

class BasePromptGenerator:
    """Base class for prompt generation techniques."""
    
    def __init__(self, llm_manager: LLMManager, data_processor: DataProcessor):
        """
        Initialize the Base Prompt Generator.
        
        Args:
            llm_manager: Instance of LLMManager for API calls
            data_processor: Instance of DataProcessor for data handling
        """
        self.llm_manager = llm_manager
        self.data_processor = data_processor
    
    def chat_completion(self, user_prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Make a chat completion request.
        
        Args:
            user_prompt: User prompt text
            system_prompt: Optional system prompt text
            
        Returns:
            LLM response text
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant developed by OpenAI that can efficiently perform tasks as per instruction"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.llm_manager.chat_completion(messages)
    
    def generate_expert_identity(self, task_description: str, expert_template: str) -> str:
        """
        Generate an expert identity description for the given task.
        
        Args:
            task_description: Description of the task
            expert_template: Template for generating expert identity
            
        Returns:
            Expert identity description
        """
        expert_prompt = expert_template.format(task_description=task_description)
        return self.chat_completion(expert_prompt)
    
    def generate_intent_keywords(self, task_description: str, instruction: str, intent_template: str) -> str:
        """
        Generate intent keywords for the given task and instruction.
        
        Args:
            task_description: Description of the task
            instruction: Instruction for solving the task
            intent_template: Template for generating intent keywords
            
        Returns:
            Intent keywords string
        """
        prompt_template = intent_template.format(
            task_description=task_description,
            instruction=instruction
        )
        return self.chat_completion(user_prompt=prompt_template)
