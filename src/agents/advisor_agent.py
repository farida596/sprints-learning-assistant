"""Advisor worker combining curriculum RAG and learner database lookups."""
from __future__ import annotations
import json, logging
from typing import Any
from uuid import uuid4
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from src.config import LEARNER_DB, MAX_RAG_REFORMULATIONS
from src.logic import best_sentences, build_placement_recommendation, context_sufficient_heuristic, is_unable_to_verify, PENDING_TAG, needs_placement, needs_profile_lookup, reformulate_query_heuristic
from src.state import SprintsState, latest_user_text, log_messages, recent_history
from src.tools.db_queries import get_db_tools, safe_get_learner_profile
from src.tools.rag_retriever import CurriculumRetriever
LOG=logging.getLogger("sprints.advisor")

class ContextGrade(BaseModel):
    """Self-reflection decision about retrieved context."""
    sufficient:bool=Field(description="True if context contains what is needed")
    reason:str=Field(description="Short explanation")

GRADE_PARSER=PydanticOutputParser(pydantic_object=ContextGrade)
GRADE_CONTEXT_PROMPT=ChatPromptTemplate.from_messages([
 ("system","Judge whether the Sprints curriculum context is sufficient to answer.\n\n{format_instructions}"),
 ("human","Question: {question}\n\nRetrieved context:\n{context}")
]).partial(format_instructions=GRADE_PARSER.get_format_instructions())
REWRITE_PROMPT=ChatPromptTemplate.from_messages([
 ("system","Rewrite the search query to retrieve better Sprints curriculum passages: expand abbreviations, name likely module topic, under 15 words. Reply with query only."),
 ("human","Original question: {question}\nPrevious query: {query}\nPrevious results were insufficient.")
])
ANSWER_PROMPT=ChatPromptTemplate.from_messages([
 ("system","You are the Sprints Advisor Agent. Answer using ONLY curriculum context and learner record. Cite module filenames in brackets. Quote salaries in EGP per month. Any tier-one placement must carry [PENDING HUMAN APPROVAL]. If information is missing, say so."),
 ("human","Earlier conversation:\n{history}\n\nQuestion: {question}\n\nLearner record: {profile}\n\nPlacement recommendation: {placement}\n\nCurriculum context:\n{context}")
])

def build_sql_react_agent(llm:Any,db_path=LEARNER_DB):
    """Create a ReAct database assistant using read-only learner tools."""
    return create_react_agent(llm,get_db_tools(db_path),prompt="You are a Sprints database assistant. Use tools to fetch learner facts. Never guess values.")

def _profile_from_messages(messages:list[Any])->dict[str,Any]:
    """Extract learner JSON from a ReAct ToolMessage."""
    for msg in messages:
        if getattr(msg,"type","")=="tool" and getattr(msg,"name","")=="get_learner_record":
            try: return json.loads(msg.content)
            except (TypeError,ValueError): continue
    return {}

def build_advisor_subgraph(llm:Any=None,retriever:CurriculumRetriever|None=None,db_path=LEARNER_DB):
    """Build the Advisor subgraph with LLM and deterministic offline paths."""
    retriever=retriever or CurriculumRetriever()
    react_agent=build_sql_react_agent(llm,db_path) if llm is not None else None
    grade_chain=(GRADE_CONTEXT_PROMPT|llm|GRADE_PARSER) if llm is not None else None
    rewrite_chain=(REWRITE_PROMPT|llm|StrOutputParser()) if llm is not None else None
    answer_chain=(ANSWER_PROMPT|llm|StrOutputParser()) if llm is not None else None
    def lookup_profile(state:SprintsState)->dict[str,Any]:
        question,learner_id=latest_user_text(state),state.get("learner_id","")
        if not needs_profile_lookup(question): return {}
        if not learner_id:
            msg=AIMessage(content="No learner id supplied; ask the learner for their SPR-#### id.",name="advisor_agent")
            log_messages("advisor.lookup",[msg]); return {"messages":[msg]}
        profile={}; new_msgs=[]
        if react_agent is not None:
            try:
                out=react_agent.invoke({"messages":[HumanMessage(content=f"Look up learner {learner_id} and report their record.")]},{"recursion_limit":8})
                new_msgs=out["messages"][1:]; profile=_profile_from_messages(new_msgs)
            except Exception as exc: LOG.warning("ReAct agent failed (%s); querying database directly",exc)
        if not profile:
            call_id=f"call_db_{uuid4().hex[:8]}"; profile=safe_get_learner_profile(learner_id,db_path)
            new_msgs=[AIMessage(content="",name="advisor_agent",tool_calls=[{"name":"get_learner_record","args":{"identifier":learner_id},"id":call_id,"type":"tool_call"}]),
                      ToolMessage(content=json.dumps(profile),tool_call_id=call_id,name="get_learner_record")]
        log_messages("advisor.lookup",new_msgs)
        update={"learner_profile":profile,"messages":new_msgs}
        if profile.get("status")=="found" and needs_placement(question): update["placement_recommendation"]=build_placement_recommendation(profile)
        return update
    def retrieve(state:SprintsState)->dict[str,Any]:
        query=state.get("search_query") or latest_user_text(state); attempt=state.get("rag_attempts",0)+1
        result=retriever.search(query,k=3); call_id=f"call_rag_{uuid4().hex[:8]}"
        summary={"query":query,"backend":result["backend"],"latency_ms":result["latency_ms"],"sources":[h["source"] for h in result["hits"]]}
        msgs=[AIMessage(content="",name="advisor_agent",tool_calls=[{"name":"search_curriculum","args":{"query":query},"id":call_id,"type":"tool_call"}]),
              ToolMessage(content=json.dumps(summary),tool_call_id=call_id,name="search_curriculum")]
        log_messages(f"advisor.retrieve#{attempt}",msgs)
        return {"retrieved_context":result["hits"],"rag_attempts":attempt,"retrieval_latency_ms":result["latency_ms"],"search_query":query,"messages":msgs}
    def grade_context(state:SprintsState)->dict[str,Any]:
        question,hits=latest_user_text(state),state.get("retrieved_context",[]); sufficient=None
        if grade_chain is not None and hits:
            try: sufficient=grade_chain.invoke({"question":question,"context":"\n---\n".join(h["text"] for h in hits)}).sufficient
            except Exception as exc: LOG.warning("Context grading failed (%s); using heuristic",exc)
        if sufficient is None: sufficient=context_sufficient_heuristic(question,hits)
        msg=AIMessage(content=f"context_sufficient={sufficient} after attempt {state.get('rag_attempts',0)}",name="advisor_agent")
        log_messages("advisor.reflect",[msg]); return {"context_sufficient":bool(sufficient),"messages":[msg]}
    def after_reflection(state:SprintsState)->str:
        return "generate" if state.get("context_sufficient") or state.get("rag_attempts",0)>MAX_RAG_REFORMULATIONS else "rewrite"
    def rewrite_query(state:SprintsState)->dict[str,Any]:
        question,old=latest_user_text(state),state.get("search_query",""); new=None
        if rewrite_chain is not None:
            try: new=rewrite_chain.invoke({"question":question,"query":old}).strip()
            except Exception as exc: LOG.warning("Query rewrite failed (%s); using heuristic",exc)
        if not new: new=reformulate_query_heuristic(question,state.get("rag_attempts",1))
        msg=AIMessage(content=f"reformulated query: {new}",name="advisor_agent")
        log_messages("advisor.rewrite",[msg]); return {"search_query":new,"messages":[msg]}
    def generate(state:SprintsState)->dict[str,Any]:
        question,hits=latest_user_text(state),state.get("retrieved_context",[])
        profile,rec=state.get("learner_profile",{}),state.get("placement_recommendation",{})
        if is_unable_to_verify(profile):
            return {"draft_answer":f"Unable to verify your record: database tool failed ({profile.get('reason')}). Please try again or contact a Sprints advisor."}
        if needs_profile_lookup(question) and not state.get("learner_id"):
            return {"draft_answer":"Please tell me your learner id (format SPR-1234) so I can look up your record."}
        if profile.get("status") in ("not_found","ambiguous"):
            return {"draft_answer":f"I could not match that learner ({profile.get('status')}). Please send your exact SPR-#### id."}
        draft=None
        if answer_chain is not None:
            try:
                draft=answer_chain.invoke({"history":recent_history(state),"question":question,
                  "profile":json.dumps(profile,ensure_ascii=False) if profile else "none",
                  "placement":json.dumps(rec,ensure_ascii=False) if rec else "none",
                  "context":"\n---\n".join(f"[{h['source']}] {h['text']}" for h in hits) or "none"}).strip()
            except Exception as exc: LOG.warning("Answer generation failed (%s); using extractive fallback",exc)
        if not draft:
            parts=[]
            if profile.get("status")=="found":
                parts.append(f"{profile['name']} ({profile['learner_id']}): {profile['completed_modules']} modules, Python score {profile['python_assignment_score']}, target salary {profile['target_salary_egp']:,} EGP/month, status {profile['readiness_status']}.")
            if rec:
                parts.append((f"Placement recommendation {PENDING_TAG}: "+", ".join(rec["recommended_employers"])+".") if rec["eligible"] else rec["reason"])
            if hits:
                facts=best_sentences(question,hits) or [hits[0]["text"][:300]]
                parts.append(f"From the curriculum [{hits[0]['source']}]: "+" ".join(facts))
            draft="\n".join(parts) or "I could not find an answer in the Sprints curriculum."
        msg=AIMessage(content=draft,name="advisor_agent"); log_messages("advisor.generate",[msg])
        return {"draft_answer":draft,"messages":[msg]}
    g=StateGraph(SprintsState)
    for name,fn in [("lookup_profile",lookup_profile),("retrieve",retrieve),("grade_context",grade_context),("rewrite",rewrite_query),("generate",generate)]: g.add_node(name,fn)
    g.add_edge(START,"lookup_profile"); g.add_edge("lookup_profile","retrieve"); g.add_edge("retrieve","grade_context")
    g.add_conditional_edges("grade_context",after_reflection,{"generate":"generate","rewrite":"rewrite"})
    g.add_edge("rewrite","retrieve"); g.add_edge("generate",END)
    return g.compile()
