"""Hand-written dev instances shared across Phase 1–3 tests.

Schema matches ``src.data.tokenize_instance``. ``chunks`` dict keys are opaque IDs,
order in the prompt comes from ``retrieval_rank`` ascending — never dict insertion.

Two fixtures:

* ``SINGLE_CHUNK_INSTANCE`` — no sys, single chunk. Used by Gate A1 to check the
  degenerate equality: with no prior context, stale KV == gold KV numerically.
* ``DEV_INSTANCE`` — full 5-chunk instance with sys + a keyword-match distractor.
  Used for Gate A2 onward, Phase 3 selection, Phase 5 single-point debug.
"""
from __future__ import annotations


# Gate A1: no sys, single chunk. chunk_0 has no prior context to attend to, so
# stale prefill == gold prefill on the same token range. fp32 tol 1e-4, bf16 tol 1e-2.
SINGLE_CHUNK_INSTANCE = {
    "sys": "",  # empty string → tokenizer yields []
    "query": "What time is lunch on Thursday?",
    "answer": "12:30 PM.",
    "chunks": {
        "chunk_0": {
            "text": (
                "Calendar — Thu Mar 14, 12:30–13:30. "
                "Lunch with Alex Chen @ Nomad Cafe, Oakland."
            ),
            "source": "calendar",
            "retrieval_rank": 1,
        },
    },
}


# Gate A2 onwards. Answer requires combining chunk_0 (when/where) + chunk_2 (what was
# discussed). chunk_1 corroborates; chunk_3 is background; chunk_4 is a keyword-match
# distractor (same Alex, same Kappa, same "lunch", but 2 months stale). No oracle
# ``is_distractor`` flag — selection must decide from K-vector divergence alone.
DEV_INSTANCE = {
    "sys": (
        "You are a helpful assistant answering questions about the "
        "user's personal data. Be concise."
    ),

    "query": "Did I have lunch with Alex last week? What did we discuss?",

    "answer": (
        "Yes, you had lunch with Alex Chen on Thursday March 14 at "
        "Nomad Cafe. You discussed the Q2 planning deck and whether "
        "to push the Project Kappa deadline from March 28 to April 4. "
        "Alex plans to send the roadmap draft by Friday."
    ),

    "chunks": {
        "chunk_0": {
            "text": (
                "Calendar event — Thu Mar 14, 12:30–13:30. "
                "Title: Lunch with Alex Chen @ Nomad Cafe. "
                "Location: 4719 Telegraph Ave, Oakland. "
                "Attendees: me, Alex Chen (accepted). "
                "Notes: bring Q2 planning deck. "
                "Created Mon Mar 11 by me. Status: confirmed."
            ),
            "source": "calendar",
            "retrieval_rank": 1,
        },
        "chunk_1": {
            "text": (
                "iMessage with Alex Chen (+1-555-0134). "
                "[Alex, Mon Mar 11 10:42] hey lunch thursday? nomad 12:30 "
                "[me, Mon 10:45] yep works "
                "[Alex, Thu 11:58] running 5 min late, bart delay "
                "[me, 12:00] np, grabbing a table "
                "[Alex, 14:47] good lunch — sending follow-up email shortly."
            ),
            "source": "messages",
            "retrieval_rank": 2,
        },
        "chunk_2": {
            "text": (
                "Email from alex.chen@nomadstudio.co, Thu Mar 14 15:22. "
                "Subject: Re: thanks for lunch. "
                "Body: Great catching up today. I'll send the full Q2 "
                "roadmap draft by EOD Friday. Main open question: keep "
                "the Project Kappa deadline on Mar 28, or push to Apr 4 "
                "to absorb the revised scope? Leaning toward the push "
                "but want your take first. — Alex"
            ),
            "source": "email",
            "retrieval_rank": 3,
        },
        "chunk_3": {
            "text": (
                "Contact — Alex Chen. Email: alex.chen@nomadstudio.co. "
                "Phone: +1-555-0134. Role: PM at Nomad Studio, lead on "
                "Project Kappa."
            ),
            "source": "contacts",
            "retrieval_rank": 4,
        },
        "chunk_4": {
            # Keyword-match distractor: retriever sees Alex + Kappa + lunch and pulls
            # this, but the date is 2 months before "last week" — irrelevant to the
            # query.
            "text": (
                "Calendar event — Fri Jan 12, 13:00–14:00. "
                "Title: Lunch with Alex Chen @ Souvla (Hayes Valley). "
                "Notes: Kappa kickoff debrief, first working lunch. "
                "Created Wed Jan 10 by me. Status: completed."
            ),
            "source": "calendar",
            "retrieval_rank": 5,
        },
    },
}
