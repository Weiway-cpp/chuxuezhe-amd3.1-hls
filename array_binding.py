# -*- coding: utf-8 -*-
"""Bind pointer-to-array-typedef parameters by hand, mechanically.

The public headers of several benchmarks declare array typedefs such as
``typedef uint8_t round_t[176];`` and pass them as ``const round_t *``.  A
parameter of that type points at the whole array, so ``key[i]`` is a whole
array, not an element, and code that indexes it does not compile.  Generated
designs and self-tests repeat that mistake; this module rewrites the invalid
form into the equivalent dereferenced form ``(*key)[i]`` without changing any
other behaviour.

The rewrite is deliberately narrow:

* only typedef names declared as arrays in the supplied header are considered;
* only identifiers declared as ``TYPE * name`` (optionally const-qualified)
  for one of those typedefs are rewritten;
* only direct ``name[`` uses are touched, so an already-correct ``(*name)[i]``
  is left alone.
"""
import re

_ARRAY_TYPEDEF = re.compile(
    r'\btypedef\s+[A-Za-z_][\w:]*\s+([A-Za-z_]\w*)\s*((?:\[\s*[^\]]*\s*\])+)\s*;')


def array_typedef_names(header):
    """Return every identifier declared as an array typedef in ``header``."""
    if not isinstance(header, str):
        return set()
    return {match.group(1) for match in _ARRAY_TYPEDEF.finditer(header)}


def bind_array_typedef_pointers(source, header):
    """Rewrite ``name[i]`` to ``(*name)[i]`` for pointer-to-array parameters.

    Returns ``(fixed_source, rewritten_names)``; when nothing matches the
    source object itself is returned unchanged.
    """
    names = array_typedef_names(header)
    if not names or not isinstance(source, str) or not source:
        return source, []
    alternation = '|'.join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    declaration = re.compile(
        r'\b(?:' + alternation + r')\s*(?:const\s*)?\*\s*(?:const\s*)?([A-Za-z_]\w*)\s*[,)=;]')
    parameters = []
    for match in declaration.finditer(source):
        name = match.group(1)
        if name not in parameters:
            parameters.append(name)
    fixed = source
    rewritten = []
    for name in parameters:
        pattern = re.compile(r'(?<![(*\w.>])' + re.escape(name) + r'\s*(?=\[)')
        replaced = pattern.sub('(*' + name + ')', fixed)
        if replaced != fixed:
            rewritten.append(name)
            fixed = replaced
    return fixed, rewritten
