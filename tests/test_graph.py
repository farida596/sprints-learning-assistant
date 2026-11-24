"""End-to-end graph tests. Skipped automatically if LangGraph is unavailable."""
import tempfile, unittest
from pathlib import Path
try:
    import langgraph
    HAVE_LANGGRAPH=True
except ImportError:
    HAVE_LANGGRAPH=False
if HAVE_LANGGRAPH:
    from src.agents.assessor_agent import build_assessor_subgraph
    from src.agents.orchestrator import build_orchestrator
    from src.config import CURRICULUM_DIR, LEARNER_CSV
    from src.main import ask, pending_interrupt, resume
    from src.tools.db_queries import build_database
    from src.tools.rag_retriever import CurriculumRetriever

@unittest.skipUnless(HAVE_LANGGRAPH,"langgraph not installed")
class OrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.db=Path(cls.tmp.name)/"learners.db"
        build_database(LEARNER_CSV,cls.db); cls.retriever=CurriculumRetriever(CURRICULUM_DIR,use_chroma=False)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def app(self): return build_orchestrator(llm=None,retriever=self.retriever,db_path=self.db)
    def test_advisor_and_assessor(self):
        out=ask(self.app(),"SPR-1013","What chunk size and overlap should I use for ingestion?")
        self.assertIn("500",out["answer"])
        grade=ask(self.app(),"SPR-1004","grade my code\n```python\nprint('hi')\n```")
        self.assertIn("Grade:",grade["answer"])
    def test_placement_interrupt_and_resume(self):
        app=self.app(); out=ask(app,"SPR-1013","Which employers can place me?")
        self.assertEqual(out["interrupt"]["type"],"placement_approval")
        done=resume(app,"SPR-1013",True,"approved")
        self.assertIn("approved this placement",done["answer"])
        self.assertIsNone(pending_interrupt(app,"SPR-1013"))
if __name__=="__main__": unittest.main()
