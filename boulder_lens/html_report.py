"""Single-file HTML report next to the annotated video. Click an event to jump to that moment."""
from __future__ import annotations

import html
import json

UI = {
    "pl": {"title": "Analiza balda", "strengths": "Co poszło dobrze", "focus": "Nad czym popracować",
           "moments": "Momenty", "limits": "Ograniczenia tej analizy", "none": "Nie wykryto błędów – świetne przejście!",
           "uncertain": "niepewne", "conf": "pewność"},
    "en": {"title": "Boulder analysis", "strengths": "What went well", "focus": "What to work on",
           "moments": "Moments", "limits": "Limits of this analysis", "none": "No issues detected – clean climb!",
           "uncertain": "uncertain", "conf": "confidence"},
}
DOT = {"bent_arms": "#eb3c3c", "foot_slip": "#ff9628", "foot_readjust": "#f0d228"}


def write_html(report: dict, path, lang: str = "pl"):
    u, fb = UI[lang], report["feedback"]
    esc = html.escape
    items = "".join(
        f'<li data-t="{e["start"]:.2f}" class="{"dim" if e["low_confidence"] else ""}">'
        f'<span class="dot" style="background:{DOT[e["kind"]]}"></span>'
        f'<span class="time">{int(e["start"] // 60)}:{int(e["start"] % 60):02d}</span>'
        f'<span>{esc(e["label"])}</span>'
        f'<span class="conf">{u["uncertain"] if e["low_confidence"] else u["conf"] + " " + str(int(e["confidence"] * 100)) + "%"}</span></li>'
        for e in report["events"])
    strengths = "".join(f"<li>{esc(s)}</li>" for s in fb["strengths"]) or "<li>–</li>"
    focus = (f'<p class="issue">{esc(fb["main_issue_label"])}</p><p>{esc(fb["main_tip"])}</p>'
             if fb["main_issue"] else f"<p>{u['none']}</p>")
    warnings = "".join(f"<li>{esc(w)}</li>" for w in fb["warnings"])
    page = f"""<!DOCTYPE html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{u['title']} – {esc(report['video'])}</title>
<style>
:root{{--bg:#f6f4ef;--card:#fff;--ink:#1d1d1b;--muted:#6b6a65;--line:#e4e1da;--accent:#2f7d5b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--card:#1f1f1d;--ink:#f1efe9;--muted:#a3a19a;--line:#33322f;--accent:#6fcf9f}}}}
*{{box-sizing:border-box}}body{{margin:0;font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--ink)}}
main{{max-width:1000px;margin:0 auto;padding:24px;display:grid;grid-template-columns:minmax(0,380px) 1fr;gap:24px}}
@media (max-width:760px){{main{{grid-template-columns:1fr}}}}
h1{{grid-column:1/-1;margin:0;font-size:22px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 8px}}
video{{width:100%;border-radius:14px;background:#000;position:sticky;top:16px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-bottom:16px}}
ul{{margin:0;padding-left:18px}} .good li::marker{{color:var(--accent)}}
.issue{{font-weight:600;font-size:18px;margin:0 0 4px}}
.moments{{list-style:none;padding:0}} .moments li{{display:flex;gap:10px;align-items:center;padding:8px 6px;border-radius:8px;cursor:pointer}}
.moments li:hover{{background:var(--line)}} .moments .dim{{opacity:.55}}
.dot{{width:10px;height:10px;border-radius:50%;flex:none}} .time{{font-variant-numeric:tabular-nums;color:var(--muted);width:38px}}
.conf{{margin-left:auto;font-size:13px;color:var(--muted)}} .limits{{font-size:14px;color:var(--muted)}}
</style></head><body><main>
<h1>{u['title']} · {esc(report['video'])}</h1>
<div><video id="v" src="{esc(report['annotated_video'])}" controls playsinline></video></div>
<div>
<section><h2>{u['strengths']}</h2><ul class="good">{strengths}</ul></section>
<section><h2>{u['focus']}</h2>{focus}</section>
{f'<section><h2>{u["moments"]}</h2><ul class="moments">{items}</ul></section>' if items else ''}
{f'<section class="limits"><h2>{u["limits"]}</h2><ul>{warnings}</ul></section>' if warnings else ''}
</div></main>
<script>
const v=document.getElementById('v');
document.querySelectorAll('.moments li').forEach(li=>li.addEventListener('click',()=>{{v.currentTime=Math.max(0,+li.dataset.t-0.5);v.play();}}));
</script>
<script type="application/json" id="data">{json.dumps({k: report[k] for k in ("metrics", "quality", "camera")}).replace("</", "<\\/")}</script>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
