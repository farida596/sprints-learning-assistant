"""LangGraph orchestrator routing learner requests to Advisor and Assessor workers."""
from __future__ import annotations
import logging
from typing import Any, Literal, Optional
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from src.agents.advisor_agent import build_advisor_subgraph
from src.agents.assessor_agent import build_assessor_subgraph
from src.config import LEARNER_DB, get_llm, llm_available
from src.logic import APPROVED_TAG, PENDING_TAG, extract_code, extract_learner_id, heuristic_intent, protocol_violations, requires_human_approval
from src.state import FINAL_SENDER, SprintsState, fresh_turn_defaults, latest_user_text, log_messages
from src.tools.rag_retriever import CurriculumRetriever
LOG=logging.getLogger("sprints.orchestrator")

class RouteDecision(BaseModel):
    """Structured intent classification."""
    intent:Literal["advisor","assessor"]=Field(description="Assessor for code or grading; otherwise advisor")
    reason:str=Field(description="Short reason")

ROUTE_PARSER=PydanticOutputParser(pydantic_object=RouteDecision)
ROUTE_PROMPT=ChatPromptTemplate.from_messages([
 ("system","Route Sprints learner messages. Advisor: curriculum, progress, salary, employers and placement. Assessor: Python code or grading requests.\n\n{format_instructions}"),
 ("human","{message}")
]).partial(format_instructions=ROUTE_PARSER.get_format_instructions())

def thread_config(learner_id:str)->dict[str,Any]:
    """Return a unique checkpointer thread configuration per learner."""
    return {"configurable":{"thread_id":f"student-{learner_id}"}}

def build_orchestrator(llm:Any="auto",retriever:Optional[CurriculumRetriever]=None,checkpointer:Any=None,db_path=LEARNER_DB):
    """Compile the multi-agent graph. Pass llm=None to force offline mode."""
    if llm=="auto": llm=get_llm() if llm_available() else None
    retriever=retriever or CurriculumRetriever()
    route_chain=(ROUTE_PROMPT|llm|ROUTE_PARSER) if llm is not None else None
    advisor=build_advisor_subgraph(llm,retriever,db_path); assessor=build_assessor_subgraph(llm)
    def intake(state:SprintsState)->dict[str,Any]:
        text=latest_user_text(state); learner_id=extract_learner_id(text) or state.get("learner_id",""); code=extract_code(text); intent=None
        if route_chain is not None:
            try: intent=route_chain.invoke({"message":text}).intent
            except Exception as exc: LOG.warning("LLM routing failed (%s); using rule-based router",exc)
        intent=intent or heuristic_intent(text)
        if code: intent="assessor"
        msg=AIMessage(content=f"orchestrator: routing learner {learner_id or '?'} to {intent}_agent",name="orchestrator")
        log_messages("orchestrator.intake",[msg])
        return {**fresh_turn_defaults(),"intent":intent,"learner_id":learner_id,"code_submission":code,"messages":[msg]}
    def route_intent(state:SprintsState)->str:
        return "assessor_agent" if state.get("intent")=="assessor" else "advisor_agent"
    def route_after_advisor(state:SprintsState)->str:
        return "human_approval" if requires_human_approval(state.get("placement_recommendation")) else "finalize"
    def human_approval(state:SprintsState)->dict[str,Any]:
        rec=state["placement_recommendation"]
        decision=interrupt({"type":"placement_approval","learner_id":rec.get("learner_id"),"learner_name":rec.get("name"),
          "tier1_employers":rec.get("tier1_employers"),"recommended_employers":rec.get("recommended_employers"),
          "target_salary_egp":rec.get("target_salary_egp"),
          "question":"Approve releasing this tier-1 placement recommendation? Reply {approved: bool, note: str}"})
        if isinstance(decision,bool): decision={"approved":decision,"note":""}
        elif not isinstance(decision,dict): decision={"approved":str(decision).strip().lower() in {"y","yes","approve","approved","true"},"note":str(decision)}
        verdict="APPROVED" if decision.get("approved") else "REJECTED"
        msg=AIMessage(content=f"human expert {verdict}: {decision.get('note','')}",name="human_expert")
        log_messages("orchestrator.hitl",[msg]); return {"human_decision":decision,"messages":[msg]}
    def finalize(state:SprintsState)->dict[str,Any]:
        answer=state.get("draft_answer","") or "Sorry, I could not produce an answer."
        decision=state.get("human_decision") or {}
        if state.get("placement_recommendation") and requires_human_approval(state.get("placement_recommendation")):
            if decision.get("approved"):
                note=f" Expert note: {decision['note']}" if decision.get("note") else ""
                answer=answer.replace(PENDING_TAG,APPROVED_TAG)+f"\n\nA Sprints career expert approved this placement recommendation.{note}"
            else:
                note=f" Expert note: {decision['note']}" if decision.get("note") else ""
                answer=("A Sprints career expert reviewed your placement recommendation and did not approve it yet."+note+
                        " Please book a session with your advisor to discuss next steps.")
        msg=AIMessage(content=answer,name=FINAL_SENDER); log_messages("orchestrator.finalize",[msg])
        for problem in protocol_violations(list(state.get("messages",[]))+[msg]): LOG.warning("agent protocol violation: %s",problem)
        return {"final_answer":answer,"messages":[msg]}
    g=StateGraph(SprintsState)
    for name,fn in [("intake",intake),("advisor_agent",advisor),("assessor_agent",assessor),("human_approval",human_approval),("finalize",finalize)]: g.add_node(name,fn)
    g.add_edge(START,"intake")
    g.add_conditional_edges("intake",route_intent,{"advisor_agent":"advisor_agent","assessor_agent":"assessor_agent"})
    g.add_conditional_edges("advisor_agent",route_after_advisor,{"human_approval":"human_approval","finalize":"finalize"})
    g.add_edge("assessor_agent","finalize"); g.add_edge("human_approval","finalize"); g.add_edge("finalize",END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
