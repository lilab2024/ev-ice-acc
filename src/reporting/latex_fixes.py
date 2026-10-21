"""Layout fixes applied to the generated LaTeX tables (called at the end of 07_make_tables_figs.py).

Every function takes and returns the LaTeX string of one table and is idempotent,
so running the script twice gives the same result.
"""
import re
from pathlib import Path


def _cells(line):
    body = line.rstrip()
    assert body.endswith("\\\\"), line
    return [c.strip() for c in body[:-2].split("&")]


def _row(cells):
    return " & ".join(cells) + " \\\\"


def steady_headway(tex):
    """Table 4: ICE range and median in separate rows, shorter headers."""
    tex = tex.replace(r"\multicolumn{2}{c}{EV bins vs. ICE range}",
                      r"\multicolumn{2}{c}{EV vs. ICE range}")
    tex = tex.replace("Below / above", "Below/above")
    out = []
    for line in tex.splitlines():
        m = re.search(r"ICE (min|max) \(\d+ vehicles\)", line)
        if not m:
            out.append(line)
            continue
        cells = _cells(line)
        lo_hi, med = [], []
        for c in cells[2:]:
            mm = re.fullmatch(r"(\S+--\S+) \((\S+)\)", c)
            lo_hi.append(mm.group(1) if mm else c)
            med.append(mm.group(2) if mm else c)
        out.append(_row([cells[0], f"ICE {m.group(1)}, range"] + lo_hi))
        out.append(_row(["", f"ICE {m.group(1)}, median"] + med))
    return "\n".join(out) + ("\n" if tex.endswith("\n") else "")


def identifiability(tex):
    """Table 7: two-level header and compact bound notation (+ upper, - lower)."""
    old = (r"Configuration & $v_0$ (m/s) & $\delta$ & $\tau$ (s) & RMSE free (m) & "
           r"RMSE fixed (m) & Change (\%) & At bound (free) \\")
    new = (r" & \multicolumn{3}{c}{Free calibration} & \multicolumn{2}{c}{RMSE (m)} & & \\" "\n"
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}" "\n"
           r"Configuration & $v_0$ (m/s) & $\delta$ & $\tau$ (s) & Free & Fixed & "
           r"Change (\%) & At bound \\")
    tex = tex.replace(old, new)

    def bounds(cell):
        conv = []
        for p in [p.strip() for p in cell.split(";")]:
            mm = re.fullmatch(r"\$(.+?)\$ (upper|lower)", p)
            if not mm:
                return cell
            sign = "+" if mm.group(2) == "upper" else "-"
            conv.append(f"${mm.group(1)}^{{{sign}}}$")
        return ", ".join(conv)

    out = []
    for line in tex.splitlines():
        if line.lstrip().startswith(r"\quad"):
            cells = _cells(line)
            cells[-1] = bounds(cells[-1])
            line = _row(cells)
        out.append(line)
    return "\n".join(out) + ("\n" if tex.endswith("\n") else "")


def minus_signs(tex):
    """Typographic minus for negative numbers written with a hyphen."""
    return re.sub(r"(?<![-\w$])-(?=\d)", r"$-$", tex)


FIXES = {
    "table2_steady_headway.tex": [steady_headway],
    "tableS3_identifiability.tex": [identifiability, minus_signs],
    "tableS2_model_pairs.tex": [minus_signs],
}


def apply_all(table_dir):
    table_dir = Path(table_dir)
    for name, funcs in FIXES.items():
        path = table_dir / name
        tex = path.read_text()
        for f in funcs:
            tex = f(tex)
        path.write_text(tex)
        print("updated", path)
