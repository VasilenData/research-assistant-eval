from resassist import assistant as A
from resassist.corpus import Paper

PAPERS = [Paper(f"2601.0000{i}", f"Title {i}", f"Abstract text number {i}.", "2026-01-01", "cs.CL") for i in range(1, 5)]


class FakeRetriever:
    def __init__(self, scores): self.scores = scores
    def search(self, q, top=4): return [(i, s) for i, s in enumerate(self.scores)][:top]


class FakeLLM:
    def __init__(self, reply): self.reply, self.calls = reply, 0
    def chat(self, system, user, max_tokens=220):
        self.calls += 1
        return self.reply


def test_prompt_numbers_sources_from_one():
    p = A.build_prompt("Why?", PAPERS[:2])
    assert "[1] Title 1." in p and "[2] Title 2." in p and p.endswith("Question: Why?")


def test_sentence_splitting_and_citations():
    s = A.split_sentences("It uses RAG [1]. It is fast [2][3]. Done [1].")
    assert len(s) == 3 and A.citations(s[1]) == [2, 3]
    assert A.strip_citations(s[1]) == "It is fast."


def test_abstention_detection():
    assert A.is_abstention(A.ABSTAIN) and A.is_abstention("I cannot answer that.")
    assert not A.is_abstention("The answer is 42 [1].")


def test_low_retrieval_score_abstains_without_calling_the_llm():
    llm = FakeLLM("should not be used [1].")
    a = A.Assistant(PAPERS, FakeRetriever([0.1, 0.05]), llm, min_score=2.0).ask("Who won the World Cup?")
    assert a.abstained and a.reason == "low retrieval score" and llm.calls == 0


def test_confident_retrieval_calls_the_llm_and_keeps_sources():
    llm = FakeLLM("It works [1].")
    a = A.Assistant(PAPERS, FakeRetriever([5.0, 1.0]), llm, min_score=2.0).ask("How?")
    assert not a.abstained and llm.calls == 1 and [p.id for p in a.sources] == [PAPERS[0].id, PAPERS[1].id]


def test_model_abstention_is_recorded():
    a = A.Assistant(PAPERS, FakeRetriever([5.0]), FakeLLM(A.ABSTAIN)).ask("?")
    assert a.abstained and a.reason == "model abstained"
