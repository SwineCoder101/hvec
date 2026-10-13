#!/usr/bin/env python3
"""Extractive payload compressor for `hvec --payload http:<url>`.

Keeps, for each passage, the sentences that share the most content words with the
question, in their original order, until a word budget is reached. A stand-in for a
learned compressor (Headroom, LLMLingua) with no dependencies beyond the standard
library; it is question-aware, which naive truncation is not.

    python3 compress_server.py 8080 60      # port, word budget per passage
    hvec query ... --payload http://127.0.0.1:8080/compress
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

STOP = set(
    "a an the of in on at to for and or but is are was were be been being by with as that this "
    "these those it its from than then there their they them we our you your he she his her not "
    "no do does did has have had can could may might will would should which what when where who "
    "how why also into over under between after before during such more most less least very".split()
)
WORD = re.compile(r"[a-z0-9]+")
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def content_words(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


def compress(question, text, budget):
    q = content_words(question)
    sentences = [s for s in SENT.split(text.strip()) if s]
    if not sentences:
        return text
    scored = []
    for i, s in enumerate(sentences):
        words = content_words(s)
        overlap = len(words & q)
        # Ties broken by position, so an all-zero passage degrades to truncation.
        scored.append((-overlap, i, s))
    scored.sort()
    kept, used = set(), 0
    matched = scored[0][0] < 0
    for neg_overlap, i, s in scored:
        # Sentences are in score order, so the first zero-overlap one means the rest
        # carry nothing for the question; keep them only when nothing matched at all.
        if neg_overlap == 0 and matched:
            break
        n = len(s.split())
        if used and used + n > budget:
            continue
        kept.add(i)
        used += n
        if used >= budget:
            break
    return " ".join(s for i, s in enumerate(sentences) if i in kept)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = [{**p, "text": compress(body["question"], p["text"], BUDGET)} for p in body["passages"]]
        data = json.dumps({"passages": out}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
