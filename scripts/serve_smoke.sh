#!/bin/sh
# Integration smoke test for `cmdrisk serve`.
#
# Covers: healthz contract, malicious/benign scoring, per-prefix answers,
# HTTP error codes (400/404/405/413), bit-identical probability parity with
# the batch CLI, and a resident-memory upper bound.
set -eu

BIN=build/cmdrisk
PORT=18080
BIND="127.0.0.1:$PORT"
BASE="http://$BIND"

if ! command -v curl >/dev/null 2>&1; then
    echo "serve_smoke: curl not found" >&2
    exit 77
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "serve_smoke: python3 not found" >&2
    exit 77
fi

tmpdir="$(mktemp -d)"
cleanup() {
    if [ -n "${SERVER_PID:-}" ]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi
    rm -rf "$tmpdir"
}
trap cleanup EXIT INT TERM

# small max-body so we can exercise the 413 path cheaply
"$BIN" serve --bind "$BIND" --max-body 8192 >"$tmpdir/server.log" 2>&1 &
SERVER_PID=$!

ready=0
i=0
while [ $i -lt 50 ]; do
    if curl -fsS --max-time 1 "$BASE/healthz" >"$tmpdir/healthz.json" 2>/dev/null; then
        ready=1
        break
    fi
    sleep 0.1
    i=$((i + 1))
done
if [ "$ready" != 1 ]; then
    echo "serve_smoke: server did not become ready" >&2
    cat "$tmpdir/server.log" >&2
    exit 1
fi

python3 - "$tmpdir/healthz.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    h = json.load(f)
assert h["ok"] is True, h
assert h["model"] == "v6", h
assert h["version"] == "v1.5.1", h
assert h["bind"].startswith("127.0.0.1:"), h
print("healthz ok")
PY

# 1) malicious single command: intent layer must force alert
curl -fsS -X POST "$BASE/v1/score" \
    -H 'Content-Type: application/json' \
    --data '{"session_id":"s1","commands":["rm -rf /"]}' \
    >"$tmpdir/score_mal.json"
python3 - "$tmpdir/score_mal.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert r["session_id"] == "s1", r
assert r["model_version"] == "v6", r
assert r["schema_version"] == "1.0", r
assert r["alerted"] is True, r
assert len(r["answers"]) == 1, r
a = r["answers"][0]
assert a["command_index"] == 0, r
assert a["alert"] is True, r
assert "destructive_delete" in a["intent_classes"], r
assert r["peak_probability"] == a["malicious_probability"], r
print("malicious scoring ok")
PY

# 2) benign command: no alert
curl -fsS -X POST "$BASE/v1/score" \
    -H 'Content-Type: application/json' \
    --data '{"commands":["echo hello world"]}' \
    >"$tmpdir/score_ok.json"
python3 - "$tmpdir/score_ok.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert r["session_id"].startswith("sess-"), r
assert r["alerted"] is False, r
a = r["answers"][0]
assert a["alert"] is False, r
assert a["intent_classes"] == "-", r
assert a["tactic"] == "-" and a["technique"] == "-", r
print("benign scoring ok")
PY

# 3) multi-command session: one answer per prefix index
curl -fsS -X POST "$BASE/v1/score" \
    -H 'Content-Type: application/json' \
    --data '{"session_id":"s2","commands":["id","whoami","rm -rf /"]}' \
    >"$tmpdir/score_multi.json"
python3 - "$tmpdir/score_multi.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert [a["command_index"] for a in r["answers"]] == [0, 1, 2], r
assert r["answers"][2]["alert"] is True, r
assert r["peak_probability"] == max(a["malicious_probability"] for a in r["answers"]), r
print("multi-command prefixes ok")
PY

# 4) dt object form is accepted
curl -fsS -X POST "$BASE/v1/score" \
    -H 'Content-Type: application/json' \
    --data '{"commands":[{"cmd":"id","dt":0.5},{"cmd":"ls /tmp","dt":null}]}' \
    >"$tmpdir/score_dt.json"
python3 - "$tmpdir/score_dt.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
assert len(r["answers"]) == 2, r
print("dt object form ok")
PY

# 5) error codes
code=$(curl -sS -o "$tmpdir/bad.json" -w '%{http_code}' -X POST "$BASE/v1/score" \
    --data '{not json')
[ "$code" = 400 ]
grep -q '"code":"invalid_json"' "$tmpdir/bad.json"

code=$(curl -sS -o "$tmpdir/empty.json" -w '%{http_code}' -X POST "$BASE/v1/score" \
    --data '{"commands":[]}')
[ "$code" = 400 ]

code=$(curl -sS -o "$tmpdir/nf.json" -w '%{http_code}' "$BASE/nope")
[ "$code" = 404 ]

code=$(curl -sS -o "$tmpdir/m405.json" -w '%{http_code}' "$BASE/v1/score")
[ "$code" = 405 ]


# 413: body larger than the 8192-byte --max-body
python3 - "$tmpdir/big.json" <<'PY'
import json, sys
with open(sys.argv[1], "w") as f:
    json.dump({"commands": ["echo " + "a" * 9000]}, f)
PY
code=$(curl -sS -o "$tmpdir/big_out.json" -w '%{http_code}' -X POST "$BASE/v1/score" \
    --data-binary @"$tmpdir/big.json")
[ "$code" = 413 ]
grep -q '"code":"payload_too_large"' "$tmpdir/big_out.json"
echo "error codes ok"

# 6) numerical parity with the batch CLI on the same binary
printf 'rm -rf /\n' >"$tmpdir/one.txt"
"$BIN" "$tmpdir/one.txt" >"$tmpdir/cli.out" 2>"$tmpdir/cli.err"
CLI_PROB=$(awk '{print $3}' "$tmpdir/cli.out")
python3 - "$tmpdir/score_mal.json" "$CLI_PROB" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    r = json.load(f)
cli = float(sys.argv[2])
http = r["answers"][0]["malicious_probability"]
assert abs(cli - http) < 1e-12, (cli, http)
assert r["answers"][0]["alert"] is True
print("CLI/HTTP probability parity ok: %.9g" % http)
PY

# 7) resident memory upper bound
python3 - "$SERVER_PID" <<'PY'
import sys
pid = sys.argv[1]
vmhwm = None
vmsize = None
with open(f"/proc/{pid}/status") as f:
    for line in f:
        if line.startswith("VmHWM:"):
            vmhwm = int(line.split()[1])
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1])
print(f"VmHWM={vmhwm} kB VmSize={vmsize} kB")
assert vmhwm is not None and vmhwm < 60 * 1024, f"VmHWM {vmhwm} kB exceeds 60 MiB"
PY

echo "serve_smoke: PASS"
