from resassist import verify
from resassist.corpus import Paper

SRC = [Paper("1", "Cats", "Cats sleep a lot.", "2026-01-01", "cs.CL"), Paper("2", "Dogs", "Dogs bark loudly.", "2026-01-01", "cs.CL")]


class FakeNLI:
    """Entails when every word of the hypothesis (minus 'a', 'lot') appears in the premise."""
    def entailment(self, pairs):
        return [1.0 if all(w in prem.lower() for w in hyp.lower().rstrip(".").split() if w not in ("a", "lot")) else 0.0
                for prem, hyp in pairs]


def test_supported_unsupported_and_invalid_citations():
    text = "Cats sleep a lot [1]. Dogs fly high [2]. Birds sing [7]. Fish swim."
    c = verify.check_answer(FakeNLI(), text, SRC)
    assert [x.supported for x in c] == [True, False, False, False]
    assert [x.valid_citation for x in c] == [True, True, False, False]


def test_guard_keeps_only_supported_sentences():
    kept, _ = verify.guard(FakeNLI(), "Cats sleep a lot [1]. Dogs fly high [2].", SRC)
    assert kept == "Cats sleep a lot [1]."


def test_guard_returns_empty_when_nothing_is_supported():
    kept, _ = verify.guard(FakeNLI(), "Dogs fly high [2].", SRC)
    assert kept == ""


def test_best_of_several_citations_counts():
    c = verify.check_answer(FakeNLI(), "Dogs bark loudly [1][2].", SRC)
    assert c[0].supported


def test_missing_citation_is_attributed_to_the_supporting_source():
    kept, c = verify.guard(FakeNLI(), "Dogs bark loudly.", SRC)
    assert kept == "Dogs bark loudly [2]."
    assert c[0].source == 2 and not c[0].valid_citation


def test_windows_include_title_and_cover_short_abstracts():
    p = Paper("3", "T", "One. Two. Three.", "2026-01-01", "cs.CL")
    w = verify.windows(p)
    assert len(w) == 2 and w[0].startswith("T. One. Two.")


def test_guard_ignores_citation_only_fragments():
    kept, _ = verify.guard(FakeNLI(), "Cats sleep a lot [1]. [1]", SRC)
    assert kept == "Cats sleep a lot [1]."


def test_revised_source_with_the_same_id_does_not_reuse_a_support_score():
    judge = FakeNLI()
    old = Paper("same", "Cats", "Cats sleep.", "2026-01-01", "cs.CL")
    revised = Paper("same", "Cats", "Cats run.", "2026-01-02", "cs.CL")
    assert verify.best_support(judge, "Cats sleep.", old) == 1
    assert verify.best_support(judge, "Cats sleep.", revised) == 0


def test_distinct_judges_do_not_share_cached_scores():
    class RejectAll:
        def entailment(self, pairs):
            return [0.] * len(pairs)
    assert verify.best_support(FakeNLI(), "Cats sleep.", SRC[0]) == 1
    assert verify.best_support(RejectAll(), "Cats sleep.", SRC[0]) == 0
