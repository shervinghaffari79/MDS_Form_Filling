"""Token features, BIO conversion and span decoding shared by the NER benchmarks. No spaCy import at module level."""


def spans_from_tokens(tokens, tags):
    """tokens: [(start, end)], tags: BIO strings. Returns [(type, start, end)]."""
    out, cur = [], None
    for (s, e), tag in zip(tokens, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[0] != tag[2:])):
            if cur:
                out.append(tuple(cur))
            cur = [tag[2:], s, e]
        elif tag.startswith("I-"):
            cur[2] = e
        else:
            if cur:
                out.append(tuple(cur))
            cur = None
    if cur:
        out.append(tuple(cur))
    return out


def token_features(doc, ner_tags, use_ner):
    feats = []
    for i, tok in enumerate(doc):
        def base(t, prefix):
            return {prefix + "lower": t.lower_, prefix + "pos": t.pos_, prefix + "tag": t.tag_, prefix + "upper": t.text.isupper(),
                    prefix + "title": t.text.istitle(), prefix + "digit": any(c.isdigit() for c in t.text), prefix + "punct": t.is_punct}
        f = {"bias": 1.0, "suf2": tok.lower_[-2:], "suf3": tok.lower_[-3:], "pre3": tok.lower_[:3], "len": min(len(tok.text), 12),
             "pct": "%" in tok.text, "first": i == 0, "last": i == len(doc) - 1, **base(tok, "")}
        if use_ner:
            f["ner"] = ner_tags[i]
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(doc):
                f.update(base(doc[j], f"{d}:"))
                if use_ner:
                    f[f"{d}:ner"] = ner_tags[j]
        feats.append(f)
    return feats


def bio_from_gold(doc, gold):
    """Tokens overlapping a gold span take its type: B- on the first token, I- on the contiguous rest."""
    tags = ["O"] * len(doc)
    for gs, ge, t in sorted(gold):
        inside = [i for i, tok in enumerate(doc) if tok.idx < ge and gs < tok.idx + len(tok)]
        for k, i in enumerate(inside):
            tags[i] = ("B-" if k == 0 else "I-") + t
    return tags
