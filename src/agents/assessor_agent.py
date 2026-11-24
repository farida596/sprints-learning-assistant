"""Assessor worker: sandboxed Python execution and evaluator-optimizer grading."""
from __future__ import annotations
import json, logging
from typing import Any, Literal
from uuid import uuid4
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator
from src.config import MAX_REFINEMENT_ITERATIONS, PASS_THRESHOLD
from src.logic import RUBRIC, heuristic_assessment, is_unable_to_verify, repair_assessment, rubric_precheck, should_refine
from src.state import SprintsState, log_messages
from src.tools.code_repl import safe_run_python
LOG=logging.getLogger("sprints.assessor")

class AssessmentResult(BaseModel):
    """Validated learner assessment."""
    score:int=Field(ge=0,le=100)
    feedback_points:list[str]=Field(min_length=1,max_length=8)
    pass_status:Literal["PASS","FAIL"]
    @field_validator("pass_status",mode="before")
    @classmethod
    def _normalise_status(cls,value:Any)->Any:
        return ("PASS" if value else "FAIL") if isinstance(value,bool) else str(value).strip().upper()

class JudgeVerdict(BaseModel):
    """Independent assessment review."""
    approved:bool
    critique:str

ASSESSMENT_PARSER=PydanticOutputParser(pydantic_object=AssessmentResult)
JUDGE_PARSER=PydanticOutputParser(pydantic_object=JudgeVerdict)
_RUBRIC_TEXT="\n".join(f"- {name}: {text}" for name,text in RUBRIC.items())
GRADE_PROMPT=ChatPromptTemplate.from_messages([
 ("system","You are the Sprints Assessor Agent grading a Module 3 Python assignment. Ground feedback in execution evidence. Rubric: printed output 50%, edge cases 20%, clarity and docstrings 20%, efficiency 10%. A score of 70+ passes. Failed code cannot pass. Be specific and actionable.\n\n{format_instructions}"),
 ("human","Learner code:\n{code}\n\nExecution stdout:\n{stdout}\n\nExecution stderr:\n{stderr}\nFlags: runtime_error={runtime_error}, timed_out={timed_out}, blocked={blocked}\nPrevious assessment: {previous}\nJudge critique: {critique}")
]).partial(format_instructions=ASSESSMENT_PARSER.get_format_instructions())
JUDGE_PROMPT=ChatPromptTemplate.from_messages([
 ("system","You are an independent grading judge. Approve only if every rubric rule is satisfied.\n"+_RUBRIC_TEXT+"\n\n{format_instructions}"),
 ("human","Execution evidence: {repl}\nAssessment: {assessment}\nAutomatic violations: {violations}")
]).partial(format_instructions=JUDGE_PARSER.get_format_instructions())

def build_assessor_subgraph(llm:Any=None):
    """Build the Assessor LangGraph subgraph; deterministic fallbacks work offline."""
    grade_chain=(GRADE_PROMPT|llm|ASSESSMENT_PARSER) if llm is not None else None
    judge_chain=(JUDGE_PROMPT|llm|JUDGE_PARSER) if llm is not None else None
    def grade_inputs(state:SprintsState,previous:str,critique:str)->dict[str,Any]:
        repl=state.get("repl_result",{})
        return {"code":state.get("code_submission",""),"stdout":repl.get("stdout",""),"stderr":repl.get("stderr",""),
                "runtime_error":repl.get("runtime_error"),"timed_out":repl.get("timed_out"),"blocked":repl.get("blocked"),
                "previous":previous,"critique":critique}
    def run_code(state:SprintsState)->dict[str,Any]:
        code=state.get("code_submission","")
        if not code.strip():
            msg=AIMessage(content="Please paste your Python code (in a ```python block) so I can run and grade it.",name="assessor_agent")
            log_messages("assessor.run_code",[msg]); return {"assessment":{"status":"needs_code"},"messages":[msg]}
        call_id=f"call_repl_{uuid4().hex[:8]}"
        request=AIMessage(content="",name="assessor_agent",tool_calls=[{"name":"run_python_code","args":{"code":code},"id":call_id,"type":"tool_call"}])
        result=safe_run_python(code); observation=ToolMessage(content=json.dumps(result),tool_call_id=call_id,name="run_python_code")
        log_messages("assessor.run_code",[request,observation]); update={"repl_result":result,"messages":[request,observation]}
        if is_unable_to_verify(result): update["assessment"]=result
        return update
    def after_run(state:SprintsState)->str:
        return "compose" if state.get("assessment",{}).get("status") in ("needs_code","Unable to verify") else "grade"
    def grade(state:SprintsState)->dict[str,Any]:
        assessment=None
        if grade_chain is not None:
            try: assessment=grade_chain.invoke(grade_inputs(state,"none","none")).model_dump()
            except Exception as exc: LOG.warning("LLM grading failed (%s); using fallback",exc)
        if assessment is None: assessment=heuristic_assessment(state.get("repl_result",{}),state.get("code_submission",""))
        msg=AIMessage(content=json.dumps(assessment),name="assessor_agent"); log_messages("assessor.grade",[msg])
        return {"assessment":assessment,"refinement_iterations":0,"messages":[msg]}
    def judge(state:SprintsState)->dict[str,Any]:
        assessment,repl=state["assessment"],state.get("repl_result",{}); violations=rubric_precheck(assessment,repl)
        llm_ok,llm_note=True,""
        if judge_chain is not None:
            try:
                verdict=judge_chain.invoke({"repl":json.dumps(repl),"assessment":json.dumps(assessment),"violations":"; ".join(violations) or "none"})
                llm_ok,llm_note=verdict.approved,verdict.critique
            except Exception as exc: LOG.warning("LLM judge failed (%s); relying on rubric",exc)
        approved=llm_ok and not violations
        result={"approved":approved,"violations":violations,"critique":"; ".join(violations) or ("" if approved else llm_note) or "ok","iteration":state.get("refinement_iterations",0)}
        msg=AIMessage(content=f"judge verdict: {json.dumps(result)}",name="judge_agent"); log_messages("assessor.judge",[msg])
        return {"judge_verdict":result,"messages":[msg]}
    def after_judge(state:SprintsState)->str:
        v=state.get("judge_verdict",{})
        return "refine" if should_refine(v.get("approved",False),state.get("refinement_iterations",0),MAX_REFINEMENT_ITERATIONS) else "compose"
    def refine(state:SprintsState)->dict[str,Any]:
        iteration=state.get("refinement_iterations",0)+1; verdict=state["judge_verdict"]; previous=state["assessment"]; refined=None
        if grade_chain is not None:
            try: refined=grade_chain.invoke(grade_inputs(state,json.dumps(previous),verdict.get("critique",""))).model_dump()
            except Exception as exc: LOG.warning("LLM refinement failed (%s); using repair",exc)
        if refined is None: refined=repair_assessment(previous,verdict.get("violations",[]),state.get("repl_result",{}))
        msg=AIMessage(content=f"refinement {iteration}: {json.dumps(refined)}",name="assessor_agent"); log_messages("assessor.refine",[msg])
        return {"assessment":refined,"refinement_iterations":iteration,"messages":[msg]}
    def compose(state:SprintsState)->dict[str,Any]:
        a=state.get("assessment",{})
        if a.get("status")=="needs_code": return {"draft_answer":"Please paste your Python code (in a ```python block) so I can run and grade it."}
        if is_unable_to_verify(a): return {"draft_answer":f"Unable to verify your submission: code execution failed ({a.get('reason')}). No grade was assigned; a human grader will review it."}
        verdict=state.get("judge_verdict",{}); lines=[f"Grade: {a['score']}/100 - {a['pass_status']}"]+[f"- {p}" for p in a["feedback_points"]]
        iters=state.get("refinement_iterations",0)
        if verdict.get("approved"): lines.append(f"(Reviewed by the judge against the rubric; {iters} refinement(s).)")
        else: lines.append(f"(The judge was still not satisfied after {MAX_REFINEMENT_ITERATIONS} refinements; flagged for human review: {verdict.get('critique')})")
        return {"draft_answer":"\n".join(lines)}
    g=StateGraph(SprintsState)
    for name,fn in [("run_code",run_code),("grade",grade),("judge",judge),("refine",refine),("compose",compose)]: g.add_node(name,fn)
    g.add_edge(START,"run_code"); g.add_conditional_edges("run_code",after_run,{"grade":"grade","compose":"compose"})
    g.add_edge("grade","judge"); g.add_conditional_edges("judge",after_judge,{"refine":"refine","compose":"compose"})
    g.add_edge("refine","judge"); g.add_edge("compose",END)
    return g.compile()
