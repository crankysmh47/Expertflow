"""Opt-in, one-process operational qualification; never a speedup study."""
from __future__ import annotations
import argparse
import http.client
import json
from pathlib import Path
import time
from expertflow.product.benchmark import write_json
from expertflow.product.profiles import load_profile
from expertflow.product.reports import freeze_sources
from expertflow.product.server import ServerSession
from expertflow.product.runtime import file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-tokens", type=int, choices=[4096, 8192], required=True)
    parser.add_argument("--chat-predict", type=int, choices=[32,256], default=32)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh output required; this protocol has no retries or resume.")
    profile = load_profile(args.profile)
    if profile["settings"]["context"] < args.long_tokens + 128:
        raise ValueError("Context must fund the real prompt and generated tokens.")
    args.output.mkdir(parents=True)
    write_json(args.output / "registration.json", {"kind":"local-operational-soak-v1", "profile":profile,
        "sources":freeze_sources(args.output), "maximum_model_processes":1,
        "maximum_generation_requests":6, "long_prompt_target_tokens":args.long_tokens,
        "chat_predict":args.chat_predict, "harness_sha256":file_digest(Path(__file__)),
        "wall_budget_seconds":300, "requests":"probe, real long prompt, two chat turns, disconnect, recovery",
        "policy":"Operational checks only; no speed, quality or global support claim.", "resume":False})
    start=time.monotonic()
    result={"status":"INCONCLUSIVE", "generation_requests":0,
            "registration_sha256":file_digest(args.output / "registration.json")}
    def remaining():
        seconds=300-(time.monotonic()-start)
        if seconds <= 0:
            raise TimeoutError("Registered wall budget exhausted.")
        return min(120,seconds)
    server=None
    try:
        with ServerSession(profile,log_dir=args.output / "session",health_timeout=remaining(),deadline=start+300) as server:
            result["generation_requests"] += 1
            write_json(args.output / "probe.json", server.complete("Say hello in one short sentence.",predict=8,timeout=remaining()))
            long_prompt=" local"*args.long_tokens
            tokenized=server.request("/tokenize",{"content":long_prompt,"add_special":True},timeout=remaining())
            if len(tokenized["tokens"]) < args.long_tokens:
                raise ValueError("Constructed prompt did not reach the frozen processed-token target.")
            result["generation_requests"] += 1
            response=server.complete(long_prompt,predict=16,timeout=remaining(),stream_measure=True)
            write_json(args.output / "long-response.json",response)
            if response["timings"]["prompt_n"] < args.long_tokens:
                raise ValueError("Runtime did not actually process the frozen long prompt.")
            messages=[{"role":"user","content":"Remember the word jade. Reply briefly."}]
            chats=[]
            for question in [None,"Which word did I ask you to remember?"]:
                if question:
                    messages.append({"role":"user","content":question})
                result["generation_requests"] += 1
                chat_start=time.monotonic()
                first=None
                chunks=[]
                for chunk in server.chat(messages,predict=args.chat_predict,timeout=remaining()):
                    if first is None:
                        first=time.monotonic()-chat_start
                    chunks.append(chunk)
                text="".join(chunks)
                if not text:
                    raise ValueError("Chat produced no text.")
                chats.append({"response":text,"ttft_seconds":first,"wall_seconds":time.monotonic()-chat_start})
                messages.append({"role":"assistant","content":text})
            write_json(args.output / "chat.json",{"turns":chats})
            result["generation_requests"] += 1
            connection=http.client.HTTPConnection("127.0.0.1",server.port,timeout=remaining())
            try:
                connection.request("POST","/completion",json.dumps({"prompt":"List many colors.","n_predict":64,"stream":True,"seed":42,"temperature":0.0}),{"Content-Type":"application/json"})
                stream=connection.getresponse()
                if stream.status != 200 or not stream.readline(1024*1024):
                    raise ValueError("Disconnect test did not establish a stream.")
            finally:
                if 'stream' in locals():
                    stream.close()
                connection.close()
            result["generation_requests"] += 1
            recovery=server.complete("Say hello.",predict=8,timeout=remaining())
            write_json(args.output / "recovery.json",recovery)
            if server.request("/health",timeout=remaining()).get("status") != "ok":
                raise ValueError("Server failed to recover after disconnect.")
            result.update(status="PASS-OPERATIONAL-SOAK",props=server.props,
                          actual_prompt_tokens=response["timings"]["prompt_n"],
                          long_ttft_seconds=response["ttft_seconds"])
        result["cleanup"]=server.child.process.poll() is not None
    except (ValueError,OSError,RuntimeError,KeyboardInterrupt) as error:
        result.update(status="INCONCLUSIVE",reason=str(error) or "Cancelled")
    finally:
        if server and server.child:
            result["cleanup"] = server.child.process.poll() is not None
        result["wall_seconds"]=time.monotonic()-start
        result["model_processes"]=len(list(args.output.glob("*/launch.json")))
        result["artifact_sha256"]={p.relative_to(args.output).as_posix():file_digest(p) for p in args.output.rglob("*") if p.is_file() and p.name != "report.json"}
        write_json(args.output / "report.json",result)
    print(json.dumps({k:result.get(k) for k in ["status","reason","model_processes","generation_requests","actual_prompt_tokens","wall_seconds","cleanup"]}))
    return 0 if result["status"] == "PASS-OPERATIONAL-SOAK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
