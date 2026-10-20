#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Teacher-forced prompt-logprob divergence between servers running the same model.

Usage (``--help`` works without a server; only ``dump`` talks to one)::

  prompt_logprob.py doc OUT.txt --tokens N [--source PATH ...]
  prompt_logprob.py dump DOC.txt OUT.json [--port 18020] [--host 127.0.0.1] [--top K]
                         [--model NAME] [--api-key-env NAME]
  prompt_logprob.py compare A.json B.json [--tail 2048]
  prompt_logprob.py refcompare NEW.json OLD.json REF.json [--tail 2048]
                         [--raw OUT.csv.gz] [--blocks 256,1024]

Subcommands
-----------
``doc`` writes a plain-text document of about N tokens. The text is the concatenation of
the files given with ``--source PATH`` (repeatable; a file is read as it is, a directory
is searched recursively for ``*.md`` and ``*.txt`` in sorted order; sources are used in
the order given). Without ``--source`` the Markdown under this repository's ``docs/`` is
used. Each file is introduced by a ``=== name ===`` line (the file name, or the path
below its source directory), and a file whose content duplicates an earlier one is
skipped. The document is cut to ``int(N * 4.2)`` characters, an estimate of the token
count; the true length is whatever the server's tokeniser reports when ``dump``
tokenises the document. The command stops with an error when the sources hold fewer
characters than that.

``dump`` tokenises DOC with the server's own ``/tokenize`` endpoint, sends the token ids
to ``/v1/completions`` with ``prompt_logprobs``, ``max_tokens: 1`` and greedy sampling,
and stores per position the logprob of the actual token, the server's top-1 token id and
its logprob. Position 0 has no logprob and is never compared. The model name is omitted
from the requests unless ``--model NAME`` or the ``VLLM_MODEL`` environment variable
gives one. The server API key is read from the environment variable named by
``--api-key-env`` (default ``VLLM_API_KEY``); no ``Authorization`` header is sent when
the variable is unset, and a variable named explicitly but unset is an error.

``dump --top K`` (K > 1) asks for ``prompt_logprobs: K`` and also stores each position's
top-K (token id, logprob) list.

``compare`` aligns two dumps (the same token ids are required) and reports, for all
positions and for the last ``--tail`` positions (those that attend over the whole
document): the mean, median, 99th percentile and maximum of |delta logprob| of the
actual token, the top-1 agreement, and the mean |delta| of the top-1 logprob. The
``all`` row starts at position 1, so the first 1,024 positions (little context, few
kernel paths in common with a long one) are included; the ``tail`` row is printed
beside it. Default thresholds: top-1 agreement >= 99.5 % and mean |delta| <= 0.02
nats in both rows; the verdict is PASS or FAIL against them and the exit status is 1
on FAIL.

Ties: vLLM's rank is 1 + the count of strictly larger logprobs, so equal logprobs
(common with bf16 logits) share rank 1 and the reported top-1 id among them is
arbitrary. With top-K dumps on both sides top-1 agreement is therefore judged on the
sets of maximal tokens (two positions agree when the sets intersect) and the strict id
rate is printed beside it; with K = 1 only the strict id comparison is possible.

When both dumps carry top-K lists, ``compare`` and ``refcompare`` add a KL divergence:
per position the distributions are coarse-grained onto the partition {each token in both
top-K sets} + {everything else}, and KL is computed on that partition in both
directions. Coarse-graining can only shrink KL (data-processing inequality), so the
printed means are LOWER BOUNDS of the true KL; the full-vocabulary KL is not available
from the API.

``refcompare`` measures two candidates against a reference dump (for example a bf16-KV
run on the FLASH_ATTN backend), each with the ``compare`` metrics (reference as the
first dump), and tests whether NEW is non-inferior to OLD:

* per position i (1 .. n-1 for ``all``, the last ``--tail`` positions for ``tail``) the
  paired difference is ``|new_i - ref_i| - |old_i - ref_i|`` of the actual-token
  logprob;
* margin = max(10 % of mean |old - ref|, 0.002 nats), the mean taken over the same
  positions;
* point verdict per row: mean |new - ref| <= mean |old - ref| + margin; the overall
  verdict (``verdict_noninferior_point`` and the exit status) is PASS when both rows
  pass;
* for each block size in ``--blocks`` (default 256 and 1024) the 95 % interval of the
  mean paired difference comes from a non-overlapping block bootstrap: the differences
  are cut into consecutive blocks of that length (the last may be shorter), whole blocks
  are resampled with replacement, the resampled mean is the sum over the picked blocks
  divided by their total length, 4000 resamples, random seed 0, and the interval is the
  2.5th and 97.5th percentile of the sorted resampled means (indices
  ``int(0.025 * 4000)`` and ``int(0.975 * 4000) - 1``).
  ``noninferior_ci_upper_block<B>`` is true when the upper end is <= margin. It is
  printed beside the point verdict and does not decide the exit status.

``--raw`` writes the per-position values as CSV.gz.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [REPO_ROOT / "docs"]
DOC_SUFFIXES = (".md", ".txt")
CHARS_PER_TOKEN = 4.2


def post(port, path, body, key=None, host="127.0.0.1", timeout=3600):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        f"http://{host}:{port}{path}", data=json.dumps(body).encode(), headers=headers
    )
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def source_files(sources):
    """(display name, path) of every text file of the sources.

    Files are taken as given; directories are searched recursively for DOC_SUFFIXES
    in sorted order.
    """
    found = []
    for s in sources:
        s = Path(s)
        if s.is_file():
            found.append((s.name, s))
        elif s.is_dir():
            found += [
                (f"{s.name}/{f.relative_to(s).as_posix()}", f)
                for f in sorted(s.rglob("*"))
                if f.is_file() and f.suffix in DOC_SUFFIXES
            ]
        else:
            raise SystemExit(f"source not found: {s}")
    return found


def build_doc(tokens: int, sources=None) -> str:
    text, seen = "", set()
    for name, f in source_files(sources if sources else DEFAULT_SOURCES):
        try:
            body = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if body in seen:
            continue
        seen.add(body)
        text += f"\n\n=== {name} ===\n{body}"
    need = int(
        tokens * CHARS_PER_TOKEN
    )  # characters; the server's own count decides the true length
    if len(text) < need:
        raise SystemExit(
            f"only {len(text)} characters of text in the sources, need about {need}; "
            "add more with --source"
        )
    return text[:need]


def api_key(env_name):
    """The key named by --api-key-env, else VLLM_API_KEY if set, else None.

    None means that no Authorization header is sent.
    """
    if env_name:
        key = os.environ.get(env_name)
        if not key:
            raise SystemExit(
                f"environment variable {env_name} (--api-key-env) is not set"
            )
        return key
    return os.environ.get("VLLM_API_KEY") or None


def dump(a):
    key = api_key(a.api_key_env)
    model = {"model": a.model} if a.model else {}
    doc = Path(a.doc).read_text(encoding="utf-8")
    ids = post(
        a.port,
        "/tokenize",
        {**model, "prompt": doc, "add_special_tokens": False},
        key=key,
        host=a.host,
    )["tokens"]
    t0 = time.time()
    r = post(
        a.port,
        "/v1/completions",
        {
            **model,
            "prompt": ids,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": a.top,
        },
        key=key,
        host=a.host,
    )
    pl = r["choices"][0]["prompt_logprobs"]
    assert len(pl) == len(ids), (len(pl), len(ids))
    actual, top1, top1lp, topk = [None], [None], [None], [None]
    for tid, ent in zip(ids[1:], pl[1:]):
        e = ent[str(tid)]
        ranked = sorted(
            ((int(k), v) for k, v in ent.items()), key=lambda kv: kv[1]["rank"]
        )
        best_id, best = ranked[0]
        actual.append(e["logprob"])
        top1.append(best_id)
        top1lp.append(best["logprob"])
        topk.append([[k, v["logprob"]] for k, v in ranked if v["rank"] <= a.top])
    out = {
        "ids": ids,
        "actual": actual,
        "top1": top1,
        "top1lp": top1lp,
        "seconds": time.time() - t0,
        "usage": r["usage"],
        "top": a.top,
    }
    if a.top > 1:
        out["topk"] = topk
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(
        f"dumped {len(ids)} positions (top {a.top}) "
        f"in {time.time() - t0:.0f}s -> {a.out}"
    )


def coarse_kl(p, q):
    """KL(p || q) and KL(q || p) on {tokens in both top-K lists} + {rest}.

    The result is a lower bound of the full-vocabulary KL.
    """
    pd, qd = dict(p), dict(q)
    common = pd.keys() & qd.keys()
    pp = [math.exp(pd[k]) for k in common]
    qq = [math.exp(qd[k]) for k in common]
    pp.append(max(1e-12, 1.0 - sum(pp)))
    qq.append(max(1e-12, 1.0 - sum(qq)))
    kl_pq = sum(x * math.log(x / y) for x, y in zip(pp, qq) if x > 0)
    kl_qp = sum(y * math.log(y / x) for x, y in zip(pp, qq) if y > 0)
    return max(0.0, kl_pq), max(0.0, kl_qp), len(common)


def argmax_set(z, i):
    return {t for t, lp in z["topk"][i] if lp == z["top1lp"][i]}


def pair_metrics(x, y, lo, n):
    """Divergence of dump y from dump x over positions lo..n-1 (see the module doc)."""
    both_topk = bool(x.get("topk") and y.get("topk"))
    d = [abs(x["actual"][i] - y["actual"][i]) for i in range(lo, n)]
    strict = sum(1 for i in range(lo, n) if x["top1"][i] == y["top1"][i]) / (n - lo)
    # vLLM ranks a token by the count of strictly larger logprobs, so tokens with EQUAL
    # logprobs (frequent: the logits are bf16) all have rank 1 and the reported "top-1"
    # id among them is arbitrary. With top-K on both sides two positions agree when
    # their sets of maximal tokens intersect; with K = 1 only the strict id comparison
    # is possible, and positions whose ids differ at an identical top-1 logprob are
    # counted.
    if both_topk:
        agree = sum(1 for i in range(lo, n) if argmax_set(x, i) & argmax_set(y, i)) / (
            n - lo
        )
    else:
        agree = strict
    d.sort()
    r = {
        "positions": n - lo,
        "mean_abs_delta": sum(d) / len(d),
        "median_abs_delta": d[len(d) // 2],
        "p99_abs_delta": d[int(0.99 * len(d))],
        "max_abs_delta": d[-1],
        "top1_agreement": agree,
        "top1_agreement_strict_ids": strict,
        "top1_tie_aware": both_topk,
        "ids_differ_at_equal_top1_logprob": sum(
            1
            for i in range(lo, n)
            if x["top1"][i] != y["top1"][i] and x["top1lp"][i] == y["top1lp"][i]
        ),
    }
    dt = [abs(x["top1lp"][i] - y["top1lp"][i]) for i in range(lo, n)]
    r["mean_abs_delta_top1_logprob"] = sum(dt) / len(dt)
    if both_topk:
        kl = [coarse_kl(x["topk"][i], y["topk"][i]) for i in range(lo, n)]
        kl_fwd = sorted(k[0] for k in kl)  # KL(x || y)
        kl_rev = sorted(k[1] for k in kl)  # KL(y || x)
        r["kl_lower_bound"] = {
            "top": min(x["top"], y["top"]),
            "mean_kl_a_b": sum(kl_fwd) / len(kl_fwd),
            "p99_kl_a_b": kl_fwd[int(0.99 * len(kl_fwd))],
            "mean_kl_b_a": sum(kl_rev) / len(kl_rev),
            "p99_kl_b_a": kl_rev[int(0.99 * len(kl_rev))],
            "mean_common_tokens": sum(k[2] for k in kl) / len(kl),
        }
    return r


def load_same(*paths):
    dumps = []
    for p in paths:
        with open(p) as f:
            dumps.append(json.load(f))
    if any(z["ids"] != dumps[0]["ids"] for z in dumps[1:]):
        raise SystemExit("token ids differ: not the same document")
    return dumps


def compare(a):
    x, y = load_same(a.a, a.b)
    n = len(x["ids"])
    rep = {
        name: pair_metrics(x, y, lo, n)
        for name, lo in (("all", 1), ("tail", max(1, n - a.tail)))
    }
    ok = all(
        r["top1_agreement"] >= 0.995 and r["mean_abs_delta"] <= 0.02
        for r in rep.values()
    )
    print(json.dumps({"n": n, **rep, "verdict": "PASS" if ok else "FAIL"}, indent=1))
    return 0 if ok else 1


def block_boot_mean(v, block, n_boot=4000, seed=0):
    """95 % CI of mean(v) by a non-overlapping block bootstrap.

    Positions of one document share their context, so neighbouring per-position
    differences are not independent; whole blocks are resampled.
    """
    blocks = [v[i : i + block] for i in range(0, len(v), block)]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        means.append(sum(sum(b) for b in pick) / sum(len(b) for b in pick))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def refcompare(a):
    """Both candidates against the reference dump; non-inferiority of NEW to OLD."""
    new, old, ref = load_same(a.new, a.old, a.ref)
    n = len(ref["ids"])
    rep = {"n": n, "new": a.new, "old": a.old, "ref": a.ref}
    for name, lo in (("all", 1), ("tail", max(1, n - a.tail))):
        mn, mo = pair_metrics(ref, new, lo, n), pair_metrics(ref, old, lo, n)
        diff = [
            abs(new["actual"][i] - ref["actual"][i])
            - abs(old["actual"][i] - ref["actual"][i])
            for i in range(lo, n)
        ]
        margin = max(0.10 * mo["mean_abs_delta"], 0.002)
        part = {
            "new_vs_ref": mn,
            "old_vs_ref": mo,
            "paired_mean_diff_new_minus_old": sum(diff) / len(diff),
            "noninferiority_margin": margin,
            "noninferior_point": mn["mean_abs_delta"] <= mo["mean_abs_delta"] + margin,
        }
        for block in a.blocks:
            lo_ci, hi_ci = block_boot_mean(diff, block)
            part[f"paired_diff_ci95_block{block}"] = [lo_ci, hi_ci]
            part[f"noninferior_ci_upper_block{block}"] = hi_ci <= margin
        rep[name] = part
    if a.raw:
        with gzip.open(a.raw, "wt") as f:
            f.write(
                "pos,token_id,lp_ref,lp_new,lp_old,"
                "absd_new_ref,absd_old_ref,diff_new_minus_old\n"
            )
            for i in range(1, n):
                dn, do = (
                    abs(new["actual"][i] - ref["actual"][i]),
                    abs(old["actual"][i] - ref["actual"][i]),
                )
                f.write(
                    f"{i},{ref['ids'][i]},{ref['actual'][i]!r},"
                    f"{new['actual'][i]!r},{old['actual'][i]!r},"
                    f"{dn!r},{do!r},{dn - do!r}\n"
                )
    ok = rep["all"]["noninferior_point"] and rep["tail"]["noninferior_point"]
    rep["verdict_noninferior_point"] = "PASS" if ok else "FAIL"
    print(json.dumps(rep, indent=1))
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(
        prog="prompt_logprob.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(
        dest="cmd", required=True, metavar="{doc,dump,compare,refcompare}"
    )
    p = sub.add_parser(
        "doc", help="build the document text from local files (no server needed)"
    )
    p.add_argument("out", help="output text file")
    p.add_argument(
        "--tokens",
        type=int,
        required=True,
        help="target token count (cut at int(N * 4.2) characters)",
    )
    p.add_argument(
        "--source",
        action="append",
        metavar="PATH",
        default=[],
        help="file or directory (searched recursively for *.md and *.txt); repeatable; "
        "default: the Markdown under this repository's docs/",
    )
    p = sub.add_parser(
        "dump", help="record per-position logprobs from a running server"
    )
    p.add_argument("doc", help="document written by 'doc'")
    p.add_argument("out", help="output JSON")
    p.add_argument("--port", type=int, default=18020)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--top",
        type=int,
        default=1,
        help="prompt_logprobs K; K > 1 also stores top-K lists",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VLLM_MODEL"),
        help="model name for the requests (default: $VLLM_MODEL, else omitted)",
    )
    p.add_argument(
        "--api-key-env",
        metavar="NAME",
        default=None,
        help="environment variable holding the server API key "
        "(default: VLLM_API_KEY if set)",
    )
    p = sub.add_parser("compare", help="two dumps against the default thresholds")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--tail", type=int, default=2048)
    p = sub.add_parser(
        "refcompare",
        help="NEW and OLD against a reference dump; non-inferiority of NEW",
    )
    p.add_argument("new")
    p.add_argument("old")
    p.add_argument("ref")
    p.add_argument("--tail", type=int, default=2048)
    p.add_argument("--raw", help="per-position CSV.gz output")
    p.add_argument(
        "--blocks",
        type=lambda s: [int(b) for b in s.split(",")],
        default=[256, 1024],
        help="comma-separated bootstrap block lengths (default 256,1024)",
    )
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "doc":
        Path(a.out).write_text(build_doc(a.tokens, a.source), encoding="utf-8")
        return 0
    if a.cmd == "dump":
        dump(a)
        return 0
    return refcompare(a) if a.cmd == "refcompare" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
