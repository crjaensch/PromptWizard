# Initialize the module
cot = SelfGeneratedCoT(api_key="your-openai-api-key")  # Or set via OPENAI_API_KEY env var

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
        "final_answer": "16π square cm"
    }
]

# Optimize the prompt with CoT
optimized_prompt, expert_identity = cot.optimize_prompt_with_cot(
    task_description, instruction, examples, answer_format
)

# Solve a new question
new_question = "If the sum of three consecutive integers is 42, what is the middle integer?"
solution = cot.solve_with_cot_prompt(optimized_prompt, expert_identity, new_question)