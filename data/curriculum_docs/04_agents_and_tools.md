# Module 8-9: Tools, ReAct Agents and Databases

## Tools
A tool is a Python function with a name, a docstring and typed arguments that an agent can call. The docstring is the tool description the model sees. Tools must validate their input and return strings or JSON that are easy for a model to read.

## ReAct pattern
ReAct stands for Reason plus Act. The agent thinks about what it needs, calls a tool, reads the observation, and repeats until it can answer. Use a ReAct agent for tasks that need external data, such as looking up a learner in the database.

## SQL tools
The learner database is a SQLite file with one table named learners. Columns are learner_id, name, bootcamp_track, completed_modules, python_assignment_score, target_salary_egp, preferred_employers and readiness_status. SQL tools must be read-only: allow SELECT statements only, reject multiple statements, and never build queries by concatenating untrusted text.

## Python REPL tool
The Python REPL tool executes learner code and returns standard output or the runtime error. Executing untrusted code is dangerous. The Sprints REPL runs code in a separate process with a time limit and memory limit, and blocks imports such as os, sys, subprocess and socket, and calls to open, eval and exec.

## Graceful failure
If a tool raises an exception, the agent returns a structured message with status "Unable to verify" and the reason, instead of guessing.
