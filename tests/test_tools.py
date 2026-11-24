"""Tests for curriculum retrieval, SQL access and the sandboxed Python REPL."""
import os, sqlite3, tempfile, unittest
from pathlib import Path
from src.config import CURRICULUM_DIR, LEARNER_CSV
from src.tools import code_repl, db_queries, rag_retriever as rag

class ChunkTests(unittest.TestCase):
    def test_overlap_and_limits(self):
        tok=rag._WhitespaceTokenizer(); chunks=rag.chunk_text(" ".join(f"w{i}" for i in range(1200)),500,50,tok)
        self.assertEqual(len(chunks),3); self.assertTrue(all(len(tok.encode(c))<=500 for c in chunks))
        self.assertEqual(chunks[0].split()[-50:],chunks[1].split()[:50])
        with self.assertRaises(ValueError): rag.chunk_text("x",50,50)
class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r=rag.CurriculumRetriever(CURRICULUM_DIR,use_chroma=False)
    def test_relevant_sources(self):
        cases={"chunk size overlap ingestion":"03_rag_and_vector_stores.md","interrupt primitive pause graph":"05_langgraph_orchestration.md",
          "tier-1 employer approval":"06_career_and_placement.md","blocked os import repl":"04_agents_and_tools.md",
          "PydanticOutputParser":"02_langchain_and_prompting.md","docstring rules functions":"01_python_foundations.md"}
        for q,source in cases.items(): self.assertIn(source,[h["source"] for h in self.r.search(q,3)["hits"]])
    def test_missing_folder(self):
        with self.assertRaises(FileNotFoundError): rag.load_curriculum_documents("/missing/folder")
class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory(); cls.db=Path(cls.tmp.name)/"test.db"; cls.count=db_queries.build_database(LEARNER_CSV,cls.db)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()
    def test_profiles_and_count(self):
        self.assertEqual(self.count,25)
        p=db_queries.get_learner_profile("spr-1013",self.db)
        self.assertEqual((p["status"],p["target_salary_egp"],p["completed_modules"]),("found",40000,12))
        self.assertEqual(p["preferred_employers"],["Fawry","Swvl","e& Egypt"])
        self.assertEqual(db_queries.get_learner_profile("Learner 13",self.db)["learner_id"],"SPR-1013")
        self.assertEqual(db_queries.get_learner_profile("Learner",self.db)["status"],"ambiguous")
    def test_select_only_and_readonly(self):
        self.assertEqual(len(db_queries.run_select("SELECT * FROM learners",self.db)),25)
        for sql in ["DELETE FROM learners","SELECT 1; DROP TABLE learners","PRAGMA table_info(learners)"]:
            with self.assertRaises(ValueError): db_queries.validate_select(sql)
        con=db_queries._connect(self.db)
        with self.assertRaises(sqlite3.OperationalError): con.execute("INSERT INTO learners VALUES ('x','x','x',1,1,1,'x','x')")
        con.close()
    def test_graceful_failure(self):
        self.assertEqual(db_queries.safe_get_learner_profile("SPR-1","/missing/db")["status"],"Unable to verify")
class ReplTests(unittest.TestCase):
    def test_execution_and_errors(self):
        self.assertEqual(code_repl.safe_run_python("print(sum(range(5)))")["stdout"].strip(),"10")
        self.assertIn("NameError",code_repl.safe_run_python("print(missing_name)")["stderr"])
        self.assertTrue(code_repl.safe_run_python("def broken(:\n pass")["runtime_error"])
    def test_security_and_timeout(self):
        for code in ["import os","from subprocess import run","open('/etc/passwd')","eval('1+1')","__import__('os')"]:
            self.assertTrue(code_repl.safe_run_python(code)["blocked"])
        self.assertTrue(code_repl.safe_run_python("while True:\n pass",1.0)["timed_out"])
    def test_environment_scrubbed(self):
        os.environ["SPRINTS_SECRET"]="private"
        self.assertEqual(code_repl.safe_run_python("print('clean')")["stdout"].strip(),"clean")
if __name__=="__main__": unittest.main()
