# Module 1-3: Python Foundations for Agent Builders

## Learning goals
Learners write clean, testable Python that agents can later wrap as tools. By the end of Module 3 a learner can define functions with type hints and docstrings, handle exceptions, work with lists, dictionaries and comprehensions, read and write files, and package code in modules.

## Functions and docstrings
Every function in the bootcamp must have a docstring that states what it does, its arguments, its return value and the errors it can raise. Agent frameworks read docstrings to decide when a tool should be called, so a vague docstring leads to wrong tool choices. Use type hints on every argument and return value.

## Error handling
Use try/except around operations that can fail, such as file access, network calls and database queries. Catch specific exceptions rather than a bare except. Return a structured error result instead of letting an exception crash the agent loop. The Sprints standard failure message for unverifiable results is "Unable to verify".

## Testing basics
Write at least one assertion-based test per function. Run snippets in isolation and capture standard output and runtime errors. Assignments in Module 3 are graded by executing the learner's code and comparing printed output with the expected output, then reviewing style, naming and docstrings.

## Assignment 3 grading rubric
Correctness of printed output is worth 50 percent. Handling of edge cases is worth 20 percent. Code clarity and docstrings are worth 20 percent. Efficiency is worth 10 percent. A score of 70 or above is a pass.
