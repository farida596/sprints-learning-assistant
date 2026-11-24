"""Framework-free decision logic shared by the agents."""
from __future__ import annotations
import re
from typing import Any, Optional
from src.config import MAX_REFINEMENT_ITERATIONS, PASS_THRESHOLD, TIER1_PARTNERS, UNABLE_TO_VERIFY

def unable_to_verify(source: str, reason: str) -> dict[str, str]:
    """Build a structured fallback when a tool fails."""
    return {"status": UNABLE_TO_VERIFY, "source": source, "reason": reason}

def is_unable_to_verify(result: Any) -> bool:
    """Return True for a structured Unable to verify result."""
    return isinstance(result, dict) and result.get("status") == UNABLE_TO_VERIFY

_LEARNER_ID_RE = re.compile(r"\bSPR-\d{4}\b", re.IGNORECASE)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ASSESSOR_HINTS = ("grade","grading","my submission","my code","review my code","assignment","check my solution","run this","test my function")
_PROFILE_HINTS = ("my progress","my modules","completed","my score","my salary","target salary","egp","my status","ready for placement","readiness","my employers","preferred employer","how am i doing","my profile")
_PLACEMENT_HINTS = ("placement","place me","job","hire","hiring","apply","employer","which company","recommend a company","fawry","swvl","vodafone","e&")

def extract_learner_id(text: str) -> Optional[str]:
    """Extract the first learner identifier."""
    match=_LEARNER_ID_RE.search(text or "")
    return match.group(0).upper() if match else None

def extract_code(text: str) -> str:
    """Extract fenced Python or a likely code block."""
    text=text or ""
    fenced=_CODE_FENCE_RE.search(text)
    if fenced: return fenced.group(1).strip("\n")
    lines=text.splitlines()
    code_start=re.compile(r"^\s*(def |class |import |from \S+ import |print\(|for |while |if |[A-Za-z_]\w*\s*=)")
    for i,line in enumerate(lines):
        if code_start.match(line): return "\n".join(lines[i:]).strip("\n")
    return ""

def heuristic_intent(text: str) -> str:
    """Route code and grading requests to assessor; all others to advisor."""
    lowered=(text or "").lower()
    return "assessor" if extract_code(text) or any(h in lowered for h in _ASSESSOR_HINTS) else "advisor"

def needs_profile_lookup(text: str) -> bool:
    """Check whether the request needs the learner record."""
    lowered=(text or "").lower()
    return any(h in lowered for h in _PROFILE_HINTS) or needs_placement(text)

def needs_placement(text: str) -> bool:
    """Check whether the request concerns jobs or placement."""
    lowered=(text or "").lower()
    return any(h in lowered for h in _PLACEMENT_HINTS)

_STOPWORDS=frozenset("a an the of to in on for and or is are was were be been do does did how what why when which who can could should would i me my we you it this that with as at by from about into than then there their them our us".split())
_ABBREVIATIONS={"rag":"retrieval-augmented generation","hitl":"human in the loop interrupt","react":"ReAct agent reason act tools","repl":"Python REPL tool","sql":"SQL database tool","llm":"language model","chroma":"ChromaDB vector store"}

def content_terms(text: str) -> list[str]:
    """Return lower-case content terms without stopwords."""
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-]+",(text or "").lower()) if w not in _STOPWORDS]

def context_sufficient_heuristic(query: str, hits: list[dict], min_coverage: float=0.5) -> bool:
    """Estimate whether retrieved chunks cover enough query terms."""
    terms=set(content_terms(query))
    if not terms or not hits: return False
    haystack=" ".join(h.get("text","") for h in hits).lower()
    return sum(1 for t in terms if t in haystack)/len(terms)>=min_coverage

def reformulate_query_heuristic(query: str, attempt: int) -> str:
    """Expand abbreviations and add curriculum terms for a retry."""
    words=re.findall(r"[A-Za-z0-9\-]+",query or "")
    expanded=[_ABBREVIATIONS.get(w.lower(),w) for w in words]
    base=" ".join(w for w in expanded if w.lower() not in _STOPWORDS)
    return base or query if attempt<=1 else f"Sprints curriculum module {base}".strip()

RUBRIC={
 "execution_grounded":"A runtime error, timeout or blocked operation must never produce a passing score.",
 "consistent_verdict":f"pass_status must be PASS only if score >= {PASS_THRESHOLD}.",
 "specific_feedback":"At least two feedback points, each specific (15+ characters).",
 "actionable":"At least one feedback point tells the learner how to fix or improve the code.",
 "error_explained":"If the code failed, feedback must name or explain the error.",
}
_ACTION_WORDS=("fix","try","use ","add ","should","consider","replace","change","avoid","remove")

def should_refine(approved: bool, iterations_done: int, max_iterations: int=MAX_REFINEMENT_ITERATIONS) -> bool:
    """Return True while the judge rejects and iteration budget remains."""
    return (not approved) and iterations_done<max_iterations

def rubric_precheck(assessment: dict, repl: dict) -> list[str]:
    """Return deterministic rubric violations for an assessment."""
    problems=[]
    score=assessment.get("score"); status=str(assessment.get("pass_status","")).upper()
    points=[str(p) for p in assessment.get("feedback_points",[])]
    failed=bool(repl.get("runtime_error") or repl.get("timed_out") or repl.get("blocked"))
    if not isinstance(score,(int,float)) or not 0<=score<=100:
        problems.append("score must be a number between 0 and 100"); return problems
    if failed and score>=PASS_THRESHOLD: problems.append(f"execution_grounded: code failed to run but score {score} >= {PASS_THRESHOLD}")
    if (score>=PASS_THRESHOLD)!=(status=="PASS"): problems.append(f"consistent_verdict: score {score} does not match pass_status {status}")
    if len([p for p in points if len(p.strip())>=15])<2: problems.append("specific_feedback: need at least two specific feedback points")
    if not any(w in p.lower() for p in points for w in _ACTION_WORDS): problems.append("actionable: no feedback point explains how to fix or improve")
    if failed and not any(any(w in p.lower() for w in ("error","fail","timeout","blocked")) for p in points): problems.append("error_explained: feedback does not mention the execution failure")
    return problems

def heuristic_assessment(repl: dict, code: str) -> dict:
    """Score code from execution evidence when no LLM is configured."""
    points=[]
    if repl.get("blocked"):
        score=10; points.append("The code used a blocked operation; remove it and use plain Python.")
    elif repl.get("timed_out"):
        score=20; points.append("Execution timed out; check for an infinite loop and try a more efficient approach.")
    elif repl.get("runtime_error"):
        score=35; last=(repl.get("stderr") or "").strip().splitlines()[-1:] or ["unknown error"]
        points.append(f"Runtime error: {last[0][:120]}. Fix this error and run the code again.")
    elif not (repl.get("stdout") or "").strip():
        score=60; points.append("The code ran without errors but printed nothing; add print statements so the output can be verified.")
    else:
        score=80; points.append("The code ran cleanly and produced output; correct execution is a good foundation.")
        points.append("Consider adding edge-case checks (empty input, negative numbers) to make the solution more robust.")
    if '"""' in code or "'''" in code:
        score=min(100,score+10); points.append("Docstrings are present, which matches the Sprints documentation standard.")
    else: points.append("Add docstrings describing arguments and return values, as required by the Sprints standard.")
    return {"score":score,"feedback_points":points,"pass_status":"PASS" if score>=PASS_THRESHOLD else "FAIL"}

def repair_assessment(assessment: dict, problems: list[str], repl: dict) -> dict:
    """Apply deterministic fixes for rubric violations."""
    fixed={"score":assessment.get("score",0),"feedback_points":list(assessment.get("feedback_points",[])),"pass_status":assessment.get("pass_status","FAIL")}
    failed=bool(repl.get("runtime_error") or repl.get("timed_out") or repl.get("blocked"))
    if failed and fixed["score"]>=PASS_THRESHOLD: fixed["score"]=min(fixed["score"],55)
    fixed["pass_status"]="PASS" if fixed["score"]>=PASS_THRESHOLD else "FAIL"
    if any(p.startswith("specific_feedback") for p in problems) or len(fixed["feedback_points"])<2:
        fixed["feedback_points"].append("Add a short comment or docstring explaining what each function returns.")
    if any(p.startswith("actionable") for p in problems):
        fixed["feedback_points"].append("You should re-run your code after fixing each issue to confirm the output.")
    if any(p.startswith("error_explained") for p in problems):
        err=((repl.get("stderr") or "").strip().splitlines() or ["execution failure"])[-1][:120]
        fixed["feedback_points"].append(f"Execution error to fix: {err}")
    return fixed

def parse_employers(raw: str | list[str] | None) -> list[str]:
    """Split employer values into a clean list."""
    if raw is None: return []
    if isinstance(raw,list): return [e.strip() for e in raw if e and e.strip()]
    return [e.strip() for e in str(raw).split(",") if e.strip()]

def is_tier1(employer: str, tier1: frozenset[str]=TIER1_PARTNERS) -> bool:
    """Check whether an employer is a tier-one partner."""
    return employer.strip().lower() in tier1

def build_placement_recommendation(profile: dict) -> dict:
    """Build a placement recommendation according to Sprints readiness policy."""
    employers=parse_employers(profile.get("preferred_employers"))
    modules=int(profile.get("completed_modules") or 0); score=float(profile.get("python_assignment_score") or 0)
    status=profile.get("readiness_status")
    eligible=status=="Ready for Placement" and modules>=12 and score>=80
    tier1=[e for e in employers if is_tier1(e)] if eligible else []
    reason=("Meets placement policy (Ready for Placement, >=12 modules, score >= 80)." if eligible else
            f"Not eligible yet: status={status}, modules={modules}/12, score={score:.0f}/80. Continue the bootcamp and meet an advisor.")
    return {"learner_id":profile.get("learner_id"),"name":profile.get("name"),"bootcamp_track":profile.get("bootcamp_track"),
            "target_salary_egp":profile.get("target_salary_egp"),"eligible":eligible,"recommended_employers":employers if eligible else [],
            "tier1_employers":tier1,"requires_human_approval":bool(tier1),"reason":reason}

def requires_human_approval(recommendation: Optional[dict]) -> bool:
    """Check whether a placement recommendation requires human approval."""
    return bool(recommendation and recommendation.get("eligible") and recommendation.get("tier1_employers"))

_TYPE_TO_CLASS={"human":"HumanMessage","ai":"AIMessage","tool":"ToolMessage","system":"SystemMessage"}

def format_protocol_message(msg: Any, max_len: int=160) -> str:
    """Render a LangChain message as one traceable log line."""
    cls=_TYPE_TO_CLASS.get(getattr(msg,"type",""),type(msg).__name__)
    sender=getattr(msg,"name",None) or "-"
    content=str(getattr(msg,"content","")).replace("\n"," ")
    if len(content)>max_len: content=content[:max_len-3]+"..."
    extras=[]; calls=getattr(msg,"tool_calls",None)
    if calls: extras.append("tool_calls="+",".join(f"{c.get('name')}#{c.get('id')}" for c in calls))
    call_id=getattr(msg,"tool_call_id",None)
    if call_id: extras.append(f"tool_call_id={call_id}")
    return f"[{cls}] from={sender} {' '.join(extras)} :: {content}".replace("  "," ")

def protocol_violations(messages: list[Any]) -> list[str]:
    """Validate agent message types, sender names and tool-call references."""
    problems=[]; open_calls=set()
    for i,msg in enumerate(messages):
        kind=getattr(msg,"type",None)
        if kind not in _TYPE_TO_CLASS: problems.append(f"message {i}: unsupported type {kind!r}")
        elif kind=="ai":
            if not getattr(msg,"name",None): problems.append(f"message {i}: AIMessage without sender name")
            for call in getattr(msg,"tool_calls",None) or []: open_calls.add(call.get("id"))
        elif kind=="tool" and getattr(msg,"tool_call_id",None) not in open_calls:
            problems.append(f"message {i}: ToolMessage {getattr(msg,'tool_call_id',None)!r} has no matching tool call")
    return problems

def best_sentences(query: str, hits: list[dict], limit: int=3) -> list[str]:
    """Select retrieved sentences with the strongest query-term overlap."""
    terms=set(content_terms(query)); scored=[]; order=0
    for hit in hits:
        body=" ".join(line for line in hit.get("text","").splitlines() if not line.lstrip().startswith("#"))
        for sentence in re.split(r"(?<=[.!?])\s+",body):
            sentence=sentence.strip()
            if len(sentence)<20: continue
            overlap=sum(1 for t in terms if t in sentence.lower())
            if overlap: scored.append((-overlap,order,sentence))
            order+=1
    scored.sort()
    return [s for _,_,s in scored[:limit]]

PENDING_TAG="[PENDING HUMAN APPROVAL]"
APPROVED_TAG="[APPROVED by a Sprints career expert]"
