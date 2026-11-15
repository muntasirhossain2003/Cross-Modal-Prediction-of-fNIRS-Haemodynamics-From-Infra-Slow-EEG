import json
import re
import sys

nb = json.load(open(sys.argv[1], encoding="utf-8"))
ansi = re.compile(r"\x1b\[[0-9;]*m")
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    outs = c.get("outputs", [])
    print(f"\n######## CELL {i} exec={c.get('execution_count')} outputs={len(outs)}")
    for o in outs:
        t = o["output_type"]
        if t == "stream":
            txt = "".join(o["text"])
        elif t == "error":
            txt = "ERROR " + o["ename"] + ": " + o["evalue"] + "\n" + "\n".join(o["traceback"][-6:])
        else:
            data = o.get("data", {})
            if "text/plain" in data:
                txt = "".join(data["text/plain"])
            else:
                txt = f"[{t}: {list(data)}]"
        txt = ansi.sub("", txt)
        if len(txt) > 6000:
            txt = txt[:3000] + "\n...[cut]...\n" + txt[-2500:]
        print(txt.rstrip())
