"""S2 answer extraction and scoring (v4, after the fourth review).

Design: the answer is **the last value the pipeline prints**, but comparison is done on a
typed prediction, and text matching is confined to the extracted answer line (never the whole
trace). Fixed rules, no reference-guided choices.

Extraction never reads the reference.
"""
import re

MISSING = {"", "-", "None", "nan", "''", '""'}
NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
ALPHA = re.compile(r"[A-Za-z\u4e00-\u9fff]")
MARKER = re.compile(r"[:：=]|\b(is|are|was|were|equals?|answer|total|difference|average)\b",
                    re.IGNORECASE)
FINAL = re.compile(r"final\s*answer\s*[:：=]?\s*", re.IGNORECASE)
NEG = re.compile(r"\b(not|no|never|false|incorrect|wrong)\b", re.IGNORECASE)
UNITS = {"million": 1e6, "billion": 1e9, "thousand": 1e3}
UNIT_RE = r"(?:%|percent|million|billion|thousand)"


def is_missing(raw):
    return str(raw).strip() in MISSING


def _clean(line):
    line = line.strip().replace("```", " ")
    had_dtype = "dtype:" in line or "Length:" in line
    line = re.sub(r"dtype:\s*\w+", " ", line)
    line = re.sub(r"Length:\s*\d+", " ", line)
    line = re.sub(r"Name:.*$", " ", line)
    for _ in range(2):
        line = re.sub(r"\s*[\(\[]\s*[^\)\]]{0,20}[\)\]][\s\.]*$", "", line)
        line = re.sub(r"\s*(?:dollars?|usd|eur|rmb|cny|yuan|euros?|pounds?|美元|元|人民币)"
                      r"[\s\.]*$", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^[\s\[\]\(\)'\"]+", "", line)
    if "[" in line and line.count("[") <= line.count("]"):
        # keep a balanced bracketed list intact ("[...np.float64(x)...]")
        line = re.sub(r"[\s'\"]+$", "", line)
    else:
        line = re.sub(r"[\s\[\]\(\)'\"]+$", "", line)
    return line.strip(), had_dtype


def _split_list(line):
    """Split an answer list on commas that are followed by whitespace.

    "2019, 2018" -> two answers; "1,234" -> one number (comma is a thousands separator).
    """
    parts = [p for p in re.split(r",\s+", line) if p.strip()]
    return parts or [line]


def _parse_value(line):
    """Return (kind, pairs) for one cleaned line, or None.

    pairs = [(value, scale), ...] -- each answer keeps its OWN scale, so a mixed output like
    "2 million, 3 thousand" yields (2, 1e6) and (3, 1e3) rather than applying one unit twice.
    """
    # bare ranges: "2019-2020", "2018/2019"
    rng = re.match(r"^\s*(?P<a>\d[\d,]*(?:\.\d+)?)\s*[-–—/]\s*(?P<b>\d[\d,]*(?:\.\d+)?)"
                   r"\s*\.?\s*$", line)
    if rng:
        return "number", [(float(rng.group("a").replace(",", "")), 1.0),
                          (float(rng.group("b").replace(",", "")), 1.0)]

    pairs = []
    # a structured list of numbers, e.g. "[np.float64(71005.0), np.float64(68490.0)]"
    for content in re.findall(r"\[([^\[\]]*)\]", line):
        c = re.sub(r"np\.float\d+\(([^)]*)\)", r"\1", content).strip().strip(",")
        if c and re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*,\s*[-+]?\d[\d,]*(?:\.\d+)?)*",
                              c):
            return "number", [(float(x.replace(",", "")), 1.0) for x in c.split(",")]
    # a line that is nothing but whitespace-separated numbers is a value list, e.g.
    # "10081 3188 10081 3188" (two answers printed twice). Series dumps are excluded by the
    # caller through the dtype/Name marker, so their index is not mistaken for an answer.
    bare_list = re.fullmatch(r"\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s+[-+]?\d[\d,]*(?:\.\d+)?)+\s*",
                             line)
    if bare_list:
        return "number", [(float(m.group(0).replace(",", "")), 1.0)
                          for m in NUM.finditer(line)]
    for part in _split_list(line):
        m = re.match(rf"^(?P<label>.*?)(?P<value>[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"
                     rf"(?P<unit>{UNIT_RE})?\s*\.?\s*$", part, re.IGNORECASE)
        if not m:
            # the part may still be a bare number inside a list like "['2019', '2018']"
            bare = re.search(r"([-+]?\d[\d,]*(?:\.\d+)?)\s*[\]'\"]?\s*$", part)
            if not bare:
                return None
            pairs.append((float(bare.group(1).replace(",", "")), 1.0))
            continue
        raw_label = m.group("label")
        label = raw_label.strip(" :=\t[]'\"")
        if label and ALPHA.search(label) and not MARKER.search(raw_label):
            return None
        unit = (m.group("unit") or "").lower()
        pairs.append((float(m.group("value").replace(",", "")), UNITS.get(unit, 1.0)))
    return ("number", pairs) if pairs else None


def extract(raw):
    """Return (kind, payload, line).

    kind: "missing" | "number" (payload = list of values, scaled by `scale`) | "text"
    For numbers the payload is a list of (value, scale) pairs so scoring can normalise units.
    """
    if is_missing(raw):
        return "missing", None, ""
    text = str(raw)

    # 1) an explicit "Final answer:" marker wins
    marks = list(FINAL.finditer(text))
    if marks:
        tail = text[marks[-1].end():]
        for line0 in reversed([x for x in tail.splitlines() if x.strip()]):
            line, _ = _clean(line0)
            got = _parse_value(line)
            if got:
                return "number", got[1], line
            return "text", line, line

    # 2) otherwise: last segment, last non-empty line
    segs = [s for s in re.split(r"Sub-query\s*\d*\s*[:：]", text) if s.strip()] or [text]
    for seg in reversed(segs):
        lines = [ln for ln in (_clean(x) for x in seg.splitlines()) if ln[0]]
        if not lines:
            continue
        line, had_dtype = lines[-1]
        got = _parse_value(line)
        if got:
            vals = got[1]
            if had_dtype and len(vals) > 1:
                vals = vals[-1:]          # a Series dump: keep the value, drop the index
            return "number", vals, line
        return "text", line, line
    return "missing", None, ""


def _is_int_text(g):
    return bool(re.fullmatch(r"[-+]?\d+", str(g).replace(",", "").strip()))


def _close(got, gold, gold_raw):
    if gold == 0:
        return abs(got) < 1e-9
    if _is_int_text(gold_raw):
        return abs(got - gold) < 1e-9
    return abs(got - gold) <= abs(gold) * 1e-4


def _gold_parts(g):
    s = str(g).replace(",", "").strip()
    if re.fullmatch(r"\d+(?:\.\d+)?\s*[-–—/]\s*\d+(?:\.\d+)?", s):
        return re.split(r"[-–—/]", s)
    return NUM.findall(s)


def _gold_scale(g):
    """Unit word attached to the reference itself, so "3.5 million" compares as 3.5e6.

    Applied symmetrically with the prediction side; otherwise a prediction that normalises
    its unit can never match a reference that carries the same unit.
    """
    low = str(g).lower()
    for w, u in UNITS.items():
        if w in low:
            return u
    return 1.0


def _text_hit(gold_text, line):
    """Whole-token containment on the answer line; a negated mention does not count.

    Word boundaries matter: "yes" must not be credited inside "yesterday".
    """
    t = gold_text.lower().strip()
    if not t:
        return False
    low = line.lower()
    pat = r"(?<!\w)" + re.escape(t) + r"(?!\w)"
    if not re.search(pat, low):
        return False
    if re.search(r"(not|no|never|false|incorrect|wrong)\s+" + re.escape(t), low):
        return False
    return True


def _num_in_text(num_text, line):
    """A numeric reference inside a text-typed prediction.

    Two labelled positions count: right after a separator ("Total podiums: 71") or at the
    very start of the answer line ("800 m"). A bare mention in running prose does not, which
    is what keeps "The question asks about 2019" rejected.
    """
    try:
        want = float(str(num_text).replace(",", ""))
    except ValueError:
        return False
    for m in re.finditer(r"[:：=]\s*([-+]?\d[\d,]*(?:\.\d+)?)", line):
        if (float(m.group(1).replace(",", "")) >= 0) == (want >= 0) and \
                _close(abs(float(m.group(1).replace(",", ""))), abs(want), num_text):
            return True
    m = re.match(r"^\s*([-+]?\d[\d,]*(?:\.\d+)?)(?![0-9.])", line)
    if m:
        got = float(m.group(1).replace(",", ""))
        return (got >= 0) == (want >= 0) and _close(abs(got), abs(want), num_text)
    return False


def score(pred, gold_list):
    kind, payload, line = pred
    if kind == "missing" or not gold_list:
        return 0.0
    hit = 0
    for g in gold_list:
        parts = _gold_parts(g)
        if parts:
            gs = _gold_scale(g)
            good = kind == "number" and all(
                any((v * sc >= 0) == (float(t) * gs >= 0)
                    and _close(abs(v * sc), abs(float(t) * gs), t)
                    for v, sc in payload)
                for t in parts)
            if not good and kind == "text":
                # only when no trailing value could be parsed: then a labelled number inside
                # the answer line is the best available reading ("800 m", "Podiums: 71")
                good = all(_num_in_text(t, line) for t in parts)
            if good:
                hit += 1
        else:
            # a text reference can sit in the same line as a numeric answer ("new york city
            # ... 1542000"), so text matching is not restricted to text-typed predictions
            if _text_hit(str(g), line):
                hit += 1
    return hit / len(gold_list)


def score_setwise(pred, gold_list):
    """Alternative metric: the answer SET must equal the reference set exactly.

    Contrast with `score`, which credits an item when every reference value appears and
    tolerates extra values (favor-recall). Signs are compared without absolute value, so this
    rule can never be more generous than `score`; `check_setwise.py` asserts that.
    """
    kind, payload, line = pred
    if kind == "missing" or not gold_list:
        return 0.0
    if kind != "number":
        return score(pred, gold_list)          # text: same containment rule, no set notion
    want = []
    for g in gold_list:
        gs = _gold_scale(g)
        want += [round(float(t) * gs, 6) for t in _gold_parts(g)]
    got = [round(v * sc, 6) for v, sc in payload]
    return 1.0 if sorted(want) == sorted(got) else 0.0


def vote_key(pred):
    kind, payload, line = pred
    if kind == "missing":
        return None
    if kind == "number":
        vs = sorted(round(v * sc, 6) for v, sc in payload)
        return "n:" + ",".join(repr(v) for v in vs)
    return "t:" + re.sub(r"\s+", " ", str(line).strip().lower())


def vote(preds):
    from collections import Counter
    pairs = [(vote_key(p), p) for p in preds]
    pairs = [(k, p) for k, p in pairs if k]
    if not pairs:
        return ("missing", None, "")
    c = Counter(k for k, _ in pairs)
    top = max(c.values())
    for k, p in pairs:
        if c[k] == top:
            return p
    return pairs[0][1]
