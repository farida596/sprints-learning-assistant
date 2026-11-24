# Module 10-12: LangGraph Orchestration and Memory

## State machines
A LangGraph StateGraph is built from a global state defined as a TypedDict, nodes that read and update the state, and edges that decide what runs next. Conditional edges choose the next node from the state, which makes dynamic routing possible. Linear chains are not used for dynamic routing in this bootcamp.

## Orchestrator-worker pattern
An orchestrator node classifies the learner intent and routes to a worker. The Advisor worker answers curriculum and career questions. The Assessor worker grades Python submissions. Workers return results to the shared state and the orchestrator finalizes the answer.

## Memory and threads
A MemorySaver checkpointer stores the state after every step. Each learner uses a unique thread id, for example the learner id, so conversations are isolated. Because the checkpointer restores the message history for the same thread id, the agent does not forget earlier turns. MemorySaver keeps data in process memory, so use a database-backed checkpointer such as SQLite or Postgres in production.

## Evaluator-optimizer
In the evaluator-optimizer pattern one node produces a result and a second LLM-as-a-judge node reviews it against an explicit rubric. If the judge rejects the result, the generator refines it using the critique. Always set a maximum number of refinement iterations; the Sprints standard is three.

## Human in the loop
The interrupt primitive pauses a graph and returns a payload to the caller. Execution resumes when the caller sends a Command with a resume value. A checkpointer is required. Use it for high-stakes steps such as approving a job placement recommendation for a tier-1 partner such as Fawry.
