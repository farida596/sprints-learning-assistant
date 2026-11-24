"""CLI entry point for the Sprints Learning Assistant."""
from __future__ import annotations
import argparse, logging, sys
from typing import Any, Optional
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from src.agents.orchestrator import build_orchestrator, thread_config
from src.config import LEARNER_CSV, LEARNER_DB, llm_available
from src.tools.db_queries import build_database

def pending_interrupt(app:Any,learner_id:str)->Optional[dict[str,Any]]:
    """Return the payload of a paused human-in-the-loop graph."""
    snapshot=app.get_state(thread_config(learner_id))
    if snapshot.next:
        for task in snapshot.tasks:
            if task.interrupts: return task.interrupts[0].value
    return None

def ask(app:Any,learner_id:str,text:str)->dict[str,Any]:
    """Send one learner message and return answer or approval interrupt."""
    config=thread_config(learner_id)
    result=app.invoke({"messages":[HumanMessage(content=text)],"learner_id":learner_id},config)
    pause=pending_interrupt(app,learner_id)
    return {"answer":None if pause else result.get("final_answer"),"interrupt":pause}

def resume(app:Any,learner_id:str,approved:bool,note:str="")->dict[str,Any]:
    """Resume a paused graph with the human expert decision."""
    result=app.invoke(Command(resume={"approved":approved,"note":note}),thread_config(learner_id))
    return {"answer":result.get("final_answer"),"interrupt":pending_interrupt(app,learner_id)}

def main(argv:Optional[list[str]]=None)->int:
    """Parse CLI options and launch the assistant."""
    parser=argparse.ArgumentParser(description="Sprints Multi-Agent AI Career & Assessment Orchestrator")
    parser.add_argument("--chat",metavar="LEARNER_ID",help="interactive chat for one learner, e.g. SPR-1013")
    parser.add_argument("--rebuild-db",action="store_true",help="rebuild SQLite from the CSV")
    parser.add_argument("--quiet",action="store_true",help="hide the agent protocol trace")
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,format="%(message)s",stream=sys.stdout)
    logging.getLogger("sprints.protocol").setLevel(logging.WARNING if args.quiet else logging.INFO)
    for noisy in ("httpx","httpcore","chromadb","openai"): logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.rebuild_db or not LEARNER_DB.exists():
        print(f"Loaded {build_database(LEARNER_CSV,LEARNER_DB)} learners into {LEARNER_DB.name}")
    print("Mode:","LLM (OpenAI)" if llm_available() else "OFFLINE (deterministic fallbacks; set OPENAI_API_KEY for the LLM)")
    app=build_orchestrator()
    if args.chat:
        learner=args.chat.upper()
        print(f"Chatting as {learner}. Type 'exit' to quit. Paste code inside ```python fences.")
        while True:
            text=input("\nYou: ").strip()
            if text.lower() in {"exit","quit"}: break
            out=ask(app,learner,text)
            if out["interrupt"]:
                print("\n[HUMAN APPROVAL NEEDED]",out["interrupt"])
                ok=input("Approve? (y/n): ").strip().lower().startswith("y")
                out=resume(app,learner,ok,input("Expert note: ").strip())
            print("\nSprints:",out["answer"])
    else: parser.print_help()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
