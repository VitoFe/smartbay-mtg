"""Chapter 4's verdict for a contrast swept over sampler seeds and issuance phases, in one place.

Every producer that prints a sweep verdict calls `verdict`, so the four outcomes mean the same thing
in every table. `verdict` returns the internal code, which the artifacts store; a table or figure
prints the label `label(code)` gives:

    code                 printed label               meaning
    result               stable                      every cell's interval excludes zero, all on one
                                                     side
    direction            sign-stable                 the seeds and the phase mean share a sign, and
                                                     every interval that excludes zero points that way
    not estimable        choice-sensitive            the seeds or the phase mean disagree in sign, or
                                                     intervals exclude zero on both sides, or an
                                                     interval excludes zero against the shared sign
    not distinguishable  no demonstrated difference  the signs agree but no interval excludes zero

Cells are (point, lo, hi) triples. `seeds` are the published grid under the sampler seeds (42 first);
`phases`, when given, are the six offsets on seed 42, of which offset 0 is the same cell as seeds[0]
and is counted once. A contrast swept over seeds only is judged on its seeds and marked so, with the
suffix ' (seeds only)', which `label` keeps.
"""

#: the printed label of each internal verdict code (review item R18); the codes stay in the artifacts
LABELS = {'result': 'stable',
          'direction': 'sign-stable',
          'not estimable': 'choice-sensitive',
          'not distinguishable': 'no demonstrated difference'}
SEEDS_ONLY = ' (seeds only)'


def label(code):
    """The printed label of a verdict code, keeping a ' (seeds only)' suffix; anything else unchanged."""
    if not isinstance(code, str):
        return code
    head, suffix = (code[:-len(SEEDS_ONLY)], SEEDS_ONLY) if code.endswith(SEEDS_ONLY) else (code, '')
    return LABELS[head] + suffix if head in LABELS else code


def _sign(x):
    return (x > 0) - (x < 0)


def excludes(lo, hi):
    return lo is not None and hi is not None and (lo > 0 or hi < 0)


def verdict(seeds, phases=None, mark_seeds_only=True):
    seeds = [tuple(c) for c in seeds]
    reference = [_sign(p) for p, _, _ in seeds]
    cells = list(seeds)
    if phases is not None:
        phases = [tuple(c) for c in phases]
        reference.append(_sign(sum(p for p, _, _ in phases) / len(phases)))
        cells += phases[1:]
    stars = {_sign(p) for p, lo, hi in cells if excludes(lo, hi)}
    if len(set(reference)) > 1 or len(stars) > 1 or (stars and stars != {reference[0]}):
        out = 'not estimable'
    elif all(excludes(lo, hi) for _, lo, hi in cells):
        out = 'result'
    elif stars:
        out = 'direction'
    else:
        out = 'not distinguishable'
    if phases is None and mark_seeds_only:
        return f'{out} (seeds only)'
    return out
