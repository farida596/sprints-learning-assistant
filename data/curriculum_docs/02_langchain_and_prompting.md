# Module 4-5: LangChain, Prompt Templates and Structured Output

## Chains and runnables
A chain composes a prompt template, a chat model and an output parser using the pipe operator. Chains are ideal for fixed, linear steps such as summarising a document or formatting feedback. When the next step depends on the content of the previous one, use a LangGraph state machine instead of a chain.

## Prompt templates
Use ChatPromptTemplate with a system message and a human message. Variables are written in braces. Keep instructions in the system message and the learner input in the human message. Partial variables can inject fixed text such as format instructions.

## Structured output with Pydantic
Define a Pydantic model for every output the system depends on. PydanticOutputParser produces format instructions for the prompt and validates the model response. If validation fails, the parser raises an error that the application must catch and handle. Field constraints, for example a score between 0 and 100, make outputs predictable and safe to store.

## Messages
LangChain uses typed messages: HumanMessage for the learner, AIMessage for the model, SystemMessage for instructions and ToolMessage for tool results. A ToolMessage must reference the tool call id of the AIMessage that requested it. Logging these messages gives a readable trace of how agents communicated.

## Temperature
Use temperature 0 for grading and routing so that results are repeatable. Higher temperatures are acceptable for brainstorming.
