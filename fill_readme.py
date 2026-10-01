"""
Fills README.template.md -> README.md from results/summary.json (and pilot/results/summary.json).

Every number in the README is written as a placeholder like
    {{main:runs.B_moe_drop.final_val:.4f}}
    {{pilot:runs.B_moe_drop.jump:+.4f}}
so no number is ever typed by hand. `--check` re-renders and fails if README.md differs
(i.e. somebody edited a number by hand, or the results changed and the README was not rebuilt).

    python fill_readme.py           # write README.md
    python fill_readme.py --check   # verify README.md matches the results on disk
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCES = {"main": os.path.join(ROOT, "results", "summary.json"),
           "pilot": os.path.join(ROOT, "pilot", "results", "summary.json")}
PAT = re.compile(r"\{\{(\w+):([\w.]+)(?::([^}]*))?\}\}")


def load():
    data = {}
    for k, p in SOURCES.items():
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data[k] = json.load(f)
    return data


def lookup(obj, dotted):
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def render(text, data):
    missing = []

    def sub(m):
        src, key, fmt = m.group(1), m.group(2), m.group(3) or ""
        if src not in data:
            missing.append(m.group(0))
            return "PENDING"
        v = lookup(data[src], key)
        if v is None:
            return "n/a"
        if fmt.endswith("%x"):                  # ratio as a percentage
            return format(v * 100, fmt[:-2]) + "%"
        return format(v, fmt)
    return PAT.sub(sub, text), missing


def main():
    with open(os.path.join(ROOT, "README.template.md"), "r", encoding="utf-8") as f:
        tpl = f.read()
    out, missing = render(tpl, load())
    readme = os.path.join(ROOT, "README.md")
    if "--check" in sys.argv:
        with open(readme, "r", encoding="utf-8") as f:
            cur = f.read()
        n = len(PAT.findall(tpl))
        if cur != out:
            print("FAIL: README.md does not match the results on disk. Run: python fill_readme.py")
            sys.exit(1)
        print(f"PASS: all {n} numbers in README.md match results on disk"
              + (f" ({len(missing)} still PENDING)" if missing else ""))
        return
    with open(readme, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"wrote README.md ({len(PAT.findall(tpl))} numbers filled, {len(missing)} PENDING)")


if __name__ == "__main__":
    main()
