"""Unit tests for framework-free decision logic."""
import unittest
from types import SimpleNamespace as NS
from src import logic as L

class RoutingTests(unittest.TestCase):
    def test_learner_id_and_code(self):
        self.assertEqual(L.extract_learner_id("I am spr-1013"),"SPR-1013")
        self.assertIsNone(L.extract_learner_id("no id"))
        self.assertEqual(L.extract_code("hi\n```python\nprint(1)\n```"),"print(1)")
        self.assertIn("def f",L.extract_code("grade\ndef f():\n return 1"))
    def test_intent(self):
        self.assertEqual(L.heuristic_intent("What is agentic RAG?"),"advisor")
        self.assertEqual(L.heuristic_intent("grade my code"),"assessor")
        self.assertEqual(L.heuristic_intent("```python\nprint(2)\n```"),"assessor")

class RagLogicTests(unittest.TestCase):
    def test_context_and_rewrite(self):
        hits=[{"text":"Chunk size is 500 tokens with 50 token overlap for ingestion."}]
        self.assertTrue(L.context_sufficient_heuristic("chunk size overlap ingestion",hits))
        self.assertFalse(L.context_sufficient_heuristic("quantum teleportation",hits))
        self.assertIn("retrieval-augmented generation",L.reformulate_query_heuristic("what is RAG",1))
        self.assertTrue(L.reformulate_query_heuristic("rag",2).startswith("Sprints curriculum module"))
    def test_best_sentences(self):
        hits=[{"text":"# Heading\nThe Sprints standard is 500 tokens per chunk with a 50 token overlap. Unrelated filler sentence about lunch."}]
        self.assertIn("500 tokens",L.best_sentences("what chunk size and overlap",hits)[0])

class AssessmentTests(unittest.TestCase):
    def test_refinement_cap(self):
        self.assertTrue(L.should_refine(False,2)); self.assertFalse(L.should_refine(False,3)); self.assertFalse(L.should_refine(True,0))
    def test_failed_code_cannot_pass(self):
        a={"score":90,"feedback_points":["Good work overall."],"pass_status":"PASS"}
        repl={"runtime_error":True,"stderr":"NameError: x"}
        self.assertTrue(any(p.startswith("execution_grounded") for p in L.rubric_precheck(a,repl)))
        fixed=L.repair_assessment(a,L.rubric_precheck(a,repl),repl)
        self.assertEqual(fixed["pass_status"],"FAIL")
    def test_heuristic_grade(self):
        a=L.heuristic_assessment({"stdout":"5\n","runtime_error":False},'def f():\n """doc"""\nprint(5)')
        self.assertEqual((a["score"],a["pass_status"]),(90,"PASS"))
        self.assertEqual(L.rubric_precheck(a,{"runtime_error":False}),[])
    def test_placement_and_hitl(self):
        profile={"learner_id":"SPR-1013","name":"Learner 13","completed_modules":12,"python_assignment_score":98,
          "readiness_status":"Ready for Placement","preferred_employers":["Fawry","Swvl"],"target_salary_egp":40000}
        rec=L.build_placement_recommendation(profile)
        self.assertTrue(rec["eligible"]); self.assertEqual(rec["tier1_employers"],["Fawry"])
        self.assertTrue(L.requires_human_approval(rec))
        rec2=L.build_placement_recommendation({**profile,"preferred_employers":"Swvl, e& Egypt"})
        self.assertFalse(L.requires_human_approval(rec2))
    def test_protocol_validation(self):
        call=NS(type="ai",name="advisor_agent",content="",tool_calls=[{"name":"t","id":"c1"}],tool_call_id=None)
        tool=NS(type="tool",name="t",content="{}",tool_calls=[],tool_call_id="c1")
        self.assertEqual(L.protocol_violations([call,tool]),[])
        self.assertEqual(len(L.protocol_violations([NS(type="tool",name="t",content="{}",tool_calls=[],tool_call_id="bad")])),1)
    def test_unable_to_verify(self):
        msg=L.unable_to_verify("sqlite_db","OperationalError")
        self.assertEqual(msg["status"],"Unable to verify"); self.assertTrue(L.is_unable_to_verify(msg))

if __name__=="__main__": unittest.main()
