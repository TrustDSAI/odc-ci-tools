"""vertex_client.py -- call Gemini models on Google Cloud Vertex AI. One file, Python 3.8+, no packages.

SETUP (once)
  1. Google Cloud project with billing, and the "Vertex AI API" (a.k.a. "Agent Platform API") enabled.
  2. Your account needs the role "Vertex AI User" on the project (owners have it).
  3. Install the Google Cloud CLI on your PC and sign in (opens the browser):
         gcloud auth application-default login
         gcloud auth application-default set-quota-project YOUR_PROJECT_ID
     This writes ~/.config/gcloud/application_default_credentials.json (a SECRET: never commit or share it).
     To run on another machine (a server), copy that file there and point GOOGLE_ADC_FILE at it.
     (Works even where the organization blocks API keys. The access token is renewed automatically.)
  4. Set your project:   export GOOGLE_CLOUD_PROJECT=your-project-id

USE
    from vertex_client import VertexClient
    c = VertexClient()                                         # project + credentials from the environment
    r = c.chat("gemini-3.8-flash", [{"role": "user", "content": "Hello"}])
    print(r["text"], r["usage"])
    r = c.chat("gemini-2.5-flash", [{"role": "system", "content": "Be brief."},
                                     {"role": "user", "content": "Hello"}], max_tokens=200)

  Model names are Vertex AI's Gemini names (e.g. gemini-3.8-flash, gemini-2.5-pro): just change the name.
  Multi-turn: append {"role": "assistant", "content": r["text"], "raw": r["raw"]} and the next user message --
  keep "raw": Gemini 3 needs its reply parts (thought signatures) sent back exactly as received.
  Optional settings (only sent when given, because models differ):
    temperature=0.7                  Gemini 3.x ignores it (Google recommends leaving it out there)
    gemini_thinking_level="LOW"      Gemini 3.x (LOW / MEDIUM / HIGH)
    gemini_thinking_budget=0         Gemini 2.5 (token budget; 0 = off where allowed)
  Test from the command line:   python3 vertex_client.py gemini-3.8-flash "Reply with just: ok"

  Transient errors (network, 429 rate limits, 5xx) are retried with back-off; anything else raises VertexError.
"""
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

TOKEN_URL = "https://oauth2.googleapis.com/token"


class VertexError(Exception):
    pass


class VertexClient:
    def __init__(self, project=None, adc_file=None, location="global", timeout=600, max_attempts=10, log=print):
        self.project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        adc_file = adc_file or os.environ.get("GOOGLE_ADC_FILE") or os.path.expanduser(
            "~/.config/gcloud/application_default_credentials.json")
        if not self.project:
            raise VertexError("no project: pass project=... or set GOOGLE_CLOUD_PROJECT")
        if not os.path.exists(adc_file):
            raise VertexError(f"credentials not found ({adc_file}): run `gcloud auth application-default login`")
        self.cred = json.load(open(adc_file))
        if not all(self.cred.get(k) for k in ("client_id", "client_secret", "refresh_token")):
            raise VertexError(f"{adc_file} is not a user sign-in file (from `gcloud auth application-default login`)")
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        self.base = f"https://{host}/v1/projects/{self.project}/locations/{location}/publishers"
        self.timeout, self.max_attempts, self.log = timeout, max_attempts, log
        self._tok, self._exp, self._lock = None, 0.0, threading.Lock()

    # ---------------------------------------------------------------- sign-in: refresh token -> 1-hour access token
    def _token(self, force=False):
        with self._lock:
            if force or not self._tok or time.time() > self._exp - 300:
                data = urllib.parse.urlencode({"grant_type": "refresh_token", "client_id": self.cred["client_id"],
                                               "client_secret": self.cred["client_secret"],
                                               "refresh_token": self.cred["refresh_token"]}).encode()
                try:
                    with urllib.request.urlopen(urllib.request.Request(TOKEN_URL, data=data), timeout=60) as r:
                        b = json.loads(r.read().decode())
                except urllib.error.HTTPError as e:
                    raise VertexError(f"sign-in refused (HTTP {e.code}): run `gcloud auth application-default login` "
                                      f"again. {e.read().decode('utf-8', 'replace')[:200]}")
                self._tok, self._exp = b["access_token"], time.time() + float(b.get("expires_in", 3600))
            return self._tok

    def _post(self, url, payload):
        refreshed = False
        for attempt in range(1, self.max_attempts + 1):
            req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST", headers={
                "Authorization": f"Bearer {self._token()}", "Content-Type": "application/json",
                "x-goog-user-project": self.project})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                code, raw, ra = e.code, e.read().decode("utf-8", "replace"), (e.headers or {}).get("Retry-After")
            except Exception as e:                                   # network error / timeout
                code, raw, ra = None, f"{type(e).__name__}: {e}", None
            if code == 401 and not refreshed:                        # token expired early: renew once
                self._token(force=True); refreshed = True
                continue
            if code is None or code == 429 or code >= 500:           # transient: wait and retry
                wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else min(60.0, 2 ** attempt)
                self.log(f"[vertex] {code or 'network'}: {raw[:120]} -- retry {attempt}/{self.max_attempts} in {wait:.0f}s")
                time.sleep(wait + random.uniform(0, 1))
                continue
            hint = {403: " (does the account have the role 'Vertex AI User' on the project?)",
                    404: " (model name / location?)"}
            raise VertexError(f"HTTP {code}{hint.get(code, '')}: {raw[:500]}")
        raise VertexError(f"gave up after {self.max_attempts} attempts")

    # ---------------------------------------------------------------- one call
    def chat(self, model, messages, max_tokens=4096, temperature=None, gemini_thinking_level=None,
             gemini_thinking_budget=None):
        """messages: [{"role": "system"|"user"|"assistant", "content": str, ("raw": from an earlier reply)}]
        -> {"text", "thinking", "stop_reason", "usage": {"input_tokens", "output_tokens"}, "raw"}"""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        gen = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            gen["temperature"] = temperature
        if gemini_thinking_level is not None:
            gen["thinkingConfig"] = {"thinkingLevel": gemini_thinking_level}
        elif gemini_thinking_budget is not None:
            gen["thinkingConfig"] = {"thinkingBudget": gemini_thinking_budget}
        p = {"generationConfig": gen, "contents": [
            {"role": "model", "parts": m.get("raw") or [{"text": m["content"]}]} if m["role"] == "assistant"
            else {"role": "user", "parts": [{"text": m["content"]}]} for m in turns]}
        if system:
            p["systemInstruction"] = {"parts": [{"text": system}]}
        b = self._post(f"{self.base}/google/models/{model}:generateContent", p)
        cand = (b.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        u = b.get("usageMetadata") or {}
        thoughts = u.get("thoughtsTokenCount", 0)                     # billed as output
        return {"text": "".join(x.get("text", "") for x in parts if not x.get("thought")),
                "thinking": "".join(x.get("text", "") for x in parts if x.get("thought")),
                "stop_reason": cand.get("finishReason") or (b.get("promptFeedback") or {}).get("blockReason"),
                "raw": parts,
                "usage": {"input_tokens": u.get("promptTokenCount", 0),
                          "output_tokens": u.get("candidatesTokenCount", 0) + thoughts}}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('usage: python3 vertex_client.py MODEL "prompt"')
    r = VertexClient().chat(sys.argv[1], [{"role": "user", "content": sys.argv[2]}], max_tokens=1024)
    print(r["text"]); print(f"[{r['stop_reason']}] {r['usage']}", file=sys.stderr)
