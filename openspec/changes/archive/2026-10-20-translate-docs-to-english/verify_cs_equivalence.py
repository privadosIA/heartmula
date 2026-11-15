#!/usr/bin/env python3
"""
Proves that a set of C# edits is cosmetic: for every changed .cs file, compares the token stream of the version in HEAD with the working tree after
removing comments and replacing the TEXT of string literals by a placeholder. Interpolation holes ({expr}, {expr,align:format}) are code and are kept.
Identifiers may differ only through the rename file (lines "old=new"). Every string literal whose text changed is printed so that it can be reviewed
by hand (a key, file name, option name, metadata key or format string must NOT appear in that list).

Usage: verify_cs_equivalence.py [--renames FILE] [--base REV] [--list] [files...]
Exit code 0 when all streams match.
"""
import re, subprocess, sys

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM = re.compile(r"[0-9][0-9A-Za-z_.]*")
OPS = re.compile(r"=>|\?\?=|\?\?|\?\.|\.\.|::|->|==|!=|<=|>=|&&|\|\||\+\+|--|<<=|>>=|<<|>>|[+\-*/%&|^]=|[^\sA-Za-z0-9_]")


class Lexer:
    def __init__(self, src):
        self.s = src
        self.i = 0
        self.tokens = []
        self.literals = []

    def run(self, until_brace=False):
        s = self.s
        depth = 0
        while self.i < len(s):
            c = s[self.i]
            n = s[self.i + 1] if self.i + 1 < len(s) else ""
            if c.isspace():
                self.i += 1
            elif c == "/" and n == "/":
                while self.i < len(s) and s[self.i] != "\n":
                    self.i += 1
            elif c == "/" and n == "*":
                j = s.index("*/", self.i + 2)
                self.i = j + 2
            elif c == '"' or (c in "$@" and self._string_start()):
                self.string()
            elif c == "'":
                j = self.i + 1
                while s[j] != "'":
                    j += 2 if s[j] == "\\" else 1
                self.tokens.append("CHAR:" + s[self.i:j + 1])
                self.i = j + 1
            elif c.isalpha() or c == "_":
                m = IDENT.match(s, self.i)
                self.tokens.append(m.group())
                self.i = m.end()
            elif c.isdigit():
                m = NUM.match(s, self.i)
                self.tokens.append(m.group())
                self.i = m.end()
            else:
                if until_brace:
                    if c == "{": depth += 1
                    if c == "}":
                        if depth == 0:
                            return
                        depth -= 1
                m = OPS.match(s, self.i)
                self.tokens.append(m.group())
                self.i = m.end()

    def _string_start(self):
        j = self.i
        while self.s[j] in "$@":
            j += 1
        return self.s[j] == '"'

    def string(self):
        s = self.s
        interp = verbatim = False
        while s[self.i] in "$@":
            interp |= s[self.i] == "$"
            verbatim |= s[self.i] == "@"
            self.i += 1
        q = 0
        while self.i + q < len(s) and s[self.i + q] == '"':
            q += 1
        text = []
        if q >= 3 and q != 2:                                   # raw string literal
            assert not interp, "interpolated raw strings are not supported"
            end = s.index('"' * q, self.i + q)
            self.literals.append(s[self.i + q:end])
            self.i = end + q
            self.tokens.append("STR")
            return
        self.i += 1
        self.tokens.append("$STR(" if interp else "STR")
        while True:
            c = s[self.i]
            n = s[self.i + 1] if self.i + 1 < len(s) else ""
            if c == '"':
                if verbatim and n == '"':
                    text.append('"'); self.i += 2; continue
                self.i += 1
                break
            if not verbatim and c == "\\":
                text.append(s[self.i:self.i + 2]); self.i += 2; continue
            if interp and c == "{":
                if n == "{":
                    text.append("{"); self.i += 2; continue
                self.i += 1
                self.tokens.append("HOLE(")
                self.hole()
                self.tokens.append(")")
                continue
            if interp and c == "}" and n == "}":
                text.append("}"); self.i += 2; continue
            text.append(c); self.i += 1
        self.literals.append("".join(text))
        if interp:
            self.tokens.append(")")

    def hole(self):
        """Tokenise the code of an interpolation hole up to its closing brace, keeping alignment/format specifiers verbatim."""
        s = self.s
        start = self.i
        depth = 0
        j = self.i
        fmt_at = None
        while True:
            c = s[j]
            if c in "([{": depth += 1
            elif c in ")]": depth -= 1
            elif c == "}":
                if depth == 0: break
                depth -= 1
            elif c == '"':                                       # nested string inside the hole
                j += 1
                while s[j] != '"':
                    j += 2 if s[j] == "\\" else 1
            elif c == ":" and depth == 0 and fmt_at is None and s[j + 1] != ":" and s[j - 1] != ":":
                fmt_at = j
            j += 1
        code_end = fmt_at if fmt_at is not None else j
        sub = Lexer(s[start:code_end])
        sub.run()
        self.tokens.extend(sub.tokens)
        self.literals.extend(sub.literals)
        if fmt_at is not None:
            self.tokens.append("FMT:" + s[fmt_at:j])
        self.i = j + 1


def lex(src):
    lx = Lexer(src)
    lx.run()
    return lx.tokens, lx.literals


def compare(old_src, new_src, renames=None):
    renames = renames or {}
    ot, ol = lex(old_src)
    nt, nl = lex(new_src)
    ot = [renames.get(t, t) for t in ot]
    problems = []
    if len(ot) != len(nt):
        problems.append(f"token count differs: {len(ot)} vs {len(nt)}")
    for k, (a, b) in enumerate(zip(ot, nt)):
        if a != b:
            problems.append(f"token {k}: {a!r} != {b!r}  (context: {' '.join(nt[max(0, k - 5):k + 3])})")
            if len(problems) > 8:
                break
    changed = []
    if len(ol) != len(nl):
        problems.append(f"string literal count differs: {len(ol)} vs {len(nl)}")
    else:
        changed = [(a, b) for a, b in zip(ol, nl) if a != b]
    return problems, changed


def main(argv):
    renames, base, show, files = {}, "HEAD", False, []
    it = iter(argv)
    for a in it:
        if a == "--renames":
            for line in open(next(it)):
                line = line.strip()
                if line and not line.startswith("#"):
                    o, n = line.split("=")
                    renames[o.strip()] = n.strip()
        elif a == "--base": base = next(it)
        elif a == "--list": show = True
        else: files.append(a)
    if not files:
        out = subprocess.run(["git", "diff", "--name-only", base, "--", "*.cs"], capture_output=True, text=True, check=True).stdout
        files = [f for f in out.split() if f.endswith(".cs")]
    bad = 0
    total_changed = 0
    for f in files:
        old = subprocess.run(["git", "show", f"{base}:{f}"], capture_output=True, text=True)
        old_src = old.stdout if old.returncode == 0 else ""
        new_src = open(f, encoding="utf-8").read()
        problems, changed = compare(old_src, new_src, renames)
        total_changed += len(changed)
        status = "OK " if not problems else "BAD"
        print(f"{status} {f}: {len(changed)} string literals changed")
        for p in problems:
            print("     " + p)
        bad += bool(problems)
        if show:
            for a, b in changed:
                print(f"     - {a!r}\n     + {b!r}")
    print(f"{len(files)} files, {bad} with problems, {total_changed} string literals changed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
