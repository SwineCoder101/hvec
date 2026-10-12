# Key compression, payload compression, and both (not yet run)

**Status: scripted, not run.** Needs a chat model and a judge (see
[2026-10-scifact-rag-two-models](../2026-10-scifact-rag-two-models/) for the same constraint).
`./run.sh` produces a 2 × 2 (or 2 × 3) grid: codec `f32` or `binary` crossed with payload stage
`none`, `truncate:60`, and optionally an external compressor at `PAYLOAD_URL`.

**Hypothesis.** Key compression changes *which* passages arrive; payload compression changes
*how much of each* arrives. If their effects on the answer rate are independent, the loss from
`binary` + `truncate` will be about the sum of the two single losses. If a smaller passage set
makes the model more sensitive to a weaker one (or the reverse), it will not.

**The external compressor cell.** `--payload http:<url>` POSTs
`{"question": "...", "passages": [{"source", "ordinal", "text"}, …]}` and expects the same
passages back with `text` replaced. A shim around a Python compressor is a few lines:

```python
# compress_server.py: python3 compress_server.py 8080
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
# from headroom import compress   # or llmlingua; any text -> shorter text function

def compress(question, text):
    return " ".join(text.split()[:60])   # stand-in: replace with the real compressor

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = [{**p, "text": compress(body["question"], p["text"])} for p in body["passages"]]
        data = json.dumps({"passages": out}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
```

**What to report.** The grid of contains and judge-correct rates with mean input tokens per
cell; the context size before and after the stage (recorded on every run); and whether
`binary` + compressor is closer to additive or to interacting.
