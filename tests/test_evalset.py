from resassist import evalset
from resassist.corpus import Paper

P = Paper("1", "Sparse retrieval for legal contracts", "We study sparse retrieval for legal contracts using BM25 variants and report recall gains on ContractNLI.", "2026-01-01", "cs.IR")


def test_parse_two_line_format():
    q, a = evalset.parse("QUESTION: How does sparse retrieval help legal contract search?\nANSWER: It raises recall on ContractNLI.")
    assert q.endswith("?") and a.startswith("It raises")
    assert evalset.parse("no format here") is None


def test_keep_rejects_banned_phrases_and_title_copies():
    ok = "How do BM25 variants improve recall for legal contracts on ContractNLI benchmark?"
    assert evalset.keep(ok, "They improve recall on the ContractNLI benchmark.", P)
    assert not evalset.keep("What does this paper say about BM25 variants for legal contracts today?", "They improve recall a lot.", P)
    assert not evalset.keep("Sparse retrieval for legal contracts?", "They improve recall a lot.", P)
    assert not evalset.keep("How good is it?", "They improve recall a lot.", P)


def test_keep_requires_rare_terms():
    df = {t: 100 for t in ("retrieval", "legal", "contracts", "sparse", "bm25", "recall")}  # all common
    q = "How do BM25 variants improve recall for legal contracts on retrieval tasks overall?"
    assert not evalset.keep(q, "They improve recall by a lot overall.", P, df, 1000)
    df.update(bm25=2, contractnli=1)
    q2 = "How do BM25 variants improve recall for legal contracts on the ContractNLI benchmark?"
    assert evalset.keep(q2, "They improve recall by a lot overall.", P, df, 1000)
