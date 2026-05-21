"""
ui_helpers.py — OrchestrAI NOC
Shared UI rendering helpers: metric cards, badges, SVG builders, Leaflet map.
"""

import math
import time
import numpy as np
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
#  COLOR / URGENCY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def urgency_color(urg: str) -> str:
    return {"Critical":"#ff6b35","Warning":"#f0b429","Monitor":"#3fb950"}.get(urg,"#7d8590")

def rul_color(rul: float) -> str:
    return "#ff6b35" if rul <= 20 else "#f0b429" if rul <= 50 else "#3fb950"

def badge(urg: str) -> str:
    cls = {"Critical":"bc","Warning":"bw","Monitor":"bm"}.get(urg,"bm")
    return f'<span class="{cls}">{urg}</span>'

def tier_html(t: str) -> str:
    return {
        "AUTO":    '<span class="ta">● AUTO</span>',
        "TIMEOUT": '<span class="tt">◑ TIMEOUT</span>',
        "HUMAN":   '<span class="th">○ HUMAN</span>',
    }.get(t, t or "")


# ─────────────────────────────────────────────────────────────────────────────
#  METRIC CARD
# ─────────────────────────────────────────────────────────────────────────────

def mc(label: str, val, sub: str = "", color: str = "var(--blue)",
       live: bool = False, tip: str = "") -> str:
    cls       = "mc-live" if live else "mc"
    tt_attr   = f'title="{tip}"' if tip else ""
    tt_style  = "cursor:help;" if tip else ""
    return (f'<div class="{cls}" {tt_attr} style="{tt_style}">'
            f'<div class="l">{label}</div>'
            f'<div class="v" style="color:{color}">{val}</div>'
            f'<div class="s">{sub}</div></div>')


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────────

def sh(label: str):
    st.markdown(f'<div class="sh">{label}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY DARK THEME DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

def pdk() -> dict:
    return dict(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        font=dict(family="IBM Plex Mono,monospace", color="#7d8590", size=10),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        margin=dict(l=36, r=16, t=36, b=36))


# ─────────────────────────────────────────────────────────────────────────────
#  SVG SPARKLINE
# ─────────────────────────────────────────────────────────────────────────────

def svg_sparkline(vals: list, color: str = "#39c5cf", W: int = 80, H: int = 28) -> str:
    if not vals or len(vals) < 2:
        return ""
    mn, mx  = min(vals), max(vals)
    rng     = max(mx - mn, 1e-9)
    pts     = " ".join(
        f"{W*i/(len(vals)-1):.1f},{H-4-(H-8)*(v-mn)/rng:.1f}"
        for i, v in enumerate(vals))
    lx = W * (len(vals)-1) / (len(vals)-1)
    ly = H - 4 - (H-8) * (vals[-1]-mn) / rng
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:{W}px;height:{H}px;display:inline-block;vertical-align:middle">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" opacity="0.9"/>'
            f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.5" fill="{color}"/></svg>')


# ─────────────────────────────────────────────────────────────────────────────
#  RUL GAUGE
# ─────────────────────────────────────────────────────────────────────────────

def svg_gauge(rul: float, cl: float, ch: float, color: str,
              W: int = 200, H: int = 130) -> str:
    cx, cy, r = 100, 105, 80
    angle = max(0, min(180, (1 - rul/125) * 180))
    rad   = math.radians(180 - angle)
    px    = cx + r * math.cos(rad)
    py    = cy - r * math.sin(rad)
    arc   = f"M {cx-r} {cy} A {r} {r} 0 0 1 {cx+r} {cy}"
    ticks = ""
    for pct, tc in [(20/125,"#ff6b35"),(50/125,"#f0b429")]:
        ta = math.radians(180 - pct*180)
        x1 = cx+(r-10)*math.cos(ta); y1 = cy-(r-10)*math.sin(ta)
        x2 = cx+(r+2)*math.cos(ta);  y2 = cy-(r+2)*math.sin(ta)
        ticks += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{tc}" stroke-width="2" opacity="0.7"/>'
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{W}px;display:block">'
            f'<path d="{arc}" fill="none" stroke="#21262d" stroke-width="14" stroke-linecap="round"/>'
            f'<path d="{arc}" fill="none" stroke="{color}" stroke-width="6" stroke-linecap="round" opacity="0.75"/>'
            f'{ticks}'
            f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
            f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="{color}"/>'
            f'<circle cx="{cx}" cy="{cy}" r="3" fill="#0d1117"/>'
            f'<text x="{cx}" y="{cy+22}" fill="{color}" font-size="22" font-weight="700" text-anchor="middle" font-family="IBM Plex Mono,monospace">{rul:.1f}</text>'
            f'<text x="{cx}" y="{cy+36}" fill="#7d8590" font-size="10" text-anchor="middle" font-family="IBM Plex Mono,monospace">cycles RUL</text>'
            f'<text x="{cx}" y="{cy+49}" fill="#7d8590" font-size="9" text-anchor="middle" font-family="IBM Plex Mono,monospace">[{cl:.1f}–{ch:.1f}]</text>'
            f'<text x="12" y="{cy+4}" fill="#ff6b35" font-size="8" font-family="monospace">20</text>'
            f'<text x="{cx-20}" y="{cy-r-6}" fill="#f0b429" font-size="8" font-family="monospace">50</text>'
            f'</svg>')


# ─────────────────────────────────────────────────────────────────────────────
#  HORIZONTAL RUL BAR CHART (SVG)
# ─────────────────────────────────────────────────────────────────────────────

def svg_rul_hbar(stations: list, live_rul_fn) -> str:
    W, ROW, PL, PR, PT = 720, 27, 168, 70, 20
    sorted_st = sorted(stations, key=live_rul_fn)
    H = PT + len(sorted_st)*ROW + 26
    bars = ""
    for i, s in enumerate(sorted_st):
        rul  = live_rul_fn(s)
        urg  = "Critical" if rul<=20 else "Warning" if rul<=50 else "Monitor"
        col  = urgency_color(urg)
        bw   = max(2, int(rul/125*(W-PL-PR)))
        y    = PT + i*ROW
        bars += (f'<text x="{PL-6}" y="{y+17}" fill="#c9d1d9" font-size="11" text-anchor="end" font-family="IBM Plex Mono,monospace">{s["id"]}</text>'
                 f'<rect x="{PL}" y="{y+4}" width="3" height="17" fill="{col}" rx="1"/>'
                 f'<rect x="{PL+5}" y="{y+5}" width="{bw}" height="15" fill="{col}" opacity="0.75" rx="2"/>'
                 f'<text x="{PL+bw+10}" y="{y+17}" fill="{col}" font-size="11" font-family="IBM Plex Mono,monospace" font-weight="700">{rul:.1f}</text>')
    for v in [20, 50, 75, 100, 125]:
        x  = PL + int(v/125*(W-PL-PR))
        tc = "#ff6b35" if v==20 else "#f0b429" if v==50 else "#1d2633"
        da = "4,3" if v<=50 else "none"
        bars += (f'<line x1="{x}" y1="{PT-4}" x2="{x}" y2="{H-20}" stroke="{tc}" stroke-width="1" stroke-dasharray="{da}" opacity="0.55"/>'
                 f'<text x="{x}" y="{H-6}" fill="#5a6475" font-size="9" text-anchor="middle">{v}</text>')
    return (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;background:#0d1117;border-radius:8px;border:1px solid #30363d">{bars}</svg>')


# ─────────────────────────────────────────────────────────────────────────────
#  LEAFLET MAP  (full standalone HTML)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_html(stations_data: list, selected_id: str) -> str:
    """
    stations_data: list of dicts with keys:
        id, urgency, rul, sub, hyp, cl, ch, conf, city, country, lat, lon
    """
    markers_js = []
    for s in stations_data:
        urg    = s["urgency"];  color = urgency_color(urg)
        sid    = s["id"];       rul   = s["rul"]
        city   = s["city"];     country = s["country"]
        sub    = s["sub"].replace("_"," ")
        hyp    = s["hyp"].replace("'","\\'").replace('"','\\"')
        conf   = s["conf"];     cl = s["cl"]; ch = s["ch"]
        is_sel = "true" if sid == selected_id else "false"
        pc     = "pulse-critical" if urg=="Critical" else "pulse-warning" if urg=="Warning" else "pulse-monitor"
        sz     = 26 if urg=="Critical" else 22 if urg=="Warning" else 18
        sz_sel = sz + 6

        # Time-to-failure from RUL and per-minute degradation rate
        degrade       = max(s.get("degrade", 0.1), 0.001)
        rem_mins      = rul / degrade
        if rem_mins < 60:
            time_to_fail = f"{rem_mins:.0f} min"
        elif rem_mins < 1440:
            h, m = int(rem_mins // 60), int(rem_mins % 60)
            time_to_fail = f"{h}h {m:02d}m"
        else:
            time_to_fail = f"{int(rem_mins // 60)}h"
        loss_per_hour = s.get("rec_loss_per_hour", s.get("rec_loss", 4200) / 6)

        markers_js.append(f"""
(function(){{
  var sid="{sid}",color="{color}",isSelected={is_sel};
  var sz=isSelected?{sz_sel}:{sz};
  var pulseIcon=L.divIcon({{className:"",iconSize:[sz,sz],iconAnchor:[sz/2,sz/2],
    html:`<div class="smark {pc} ${{isSelected?'smark-sel':''}}"
      style="width:${{sz}}px;height:${{sz}}px;background:${{color}};
      border:${{isSelected?'3px solid #fff':'2px solid rgba(255,255,255,0.45)'}};
      border-radius:50%;cursor:pointer;"></div>`
  }});
  var m=L.marker([{s['lat']},{s['lon']}],{{icon:pulseIcon,zIndexOffset:{300 if urg=="Critical" else 200 if urg=="Warning" else 100}}});
  m.addTo(map);
  m.bindPopup(`
    <div style="font-family:'IBM Plex Mono',monospace;min-width:240px;max-width:268px;background:#161b22;color:#e6edf3;border-radius:8px;overflow:hidden">
      <div style="background:{color}18;border-left:4px solid {color};padding:8px 12px">
        <div style="font-size:.95rem;font-weight:700;color:{color}">{sid}</div>
        <div style="font-size:.60rem;color:#7d8590;margin-top:2px">📍 {city}, {country}</div>
      </div>
      <div style="padding:9px 12px">

        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span style="font-size:.60rem;color:#7d8590">URGENCY</span>
          <span style="font-size:.63rem;font-weight:700;color:{color};background:{color}22;padding:1px 7px;border-radius:3px">{urg.upper()}</span>
        </div>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span style="font-size:.60rem;color:#7d8590">LIVE RUL</span>
          <span style="font-size:.75rem;font-weight:700;color:{color}">{rul:.1f} cycles</span>
        </div>

        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
          <span style="font-size:.60rem;color:#7d8590">TIME TO FAILURE</span>
          <span style="font-size:.72rem;font-weight:700;color:{color}">{time_to_fail}</span>
        </div>

        <div style="background:#21262d;height:2px;border-radius:2px;margin-bottom:7px">
          <div style="width:{min(100,int(rul/125*100))}%;height:2px;background:{color};border-radius:2px"></div>
        </div>

        <div style="background:#ff6b3512;border-radius:4px;padding:5px 8px;margin-bottom:5px">
          <div style="font-size:.57rem;color:#ff6b35;font-weight:700;margin-bottom:2px">💶 FINANCIAL EXPOSURE</div>
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:.59rem;color:#7d8590">est. total loss</span>
            <span style="font-size:.68rem;font-weight:700;color:#ff6b35">€{s.get('rec_loss',4200):,.0f}</span>
          </div>
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:.59rem;color:#7d8590">loss / hour</span>
            <span style="font-size:.63rem;color:#f0b429">€{loss_per_hour:,.0f}</span>
          </div>
        </div>
        <div style="background:#39c5cf12;border-radius:4px;padding:5px 8px;margin-bottom:5px">
          <div style="font-size:.57rem;color:#39c5cf;font-weight:700;margin-bottom:2px">🔧 EST. REPAIR TIME</div>
          <div style="display:flex;justify-content:space-between">
            <span style="font-size:.59rem;color:#7d8590">travel + fix + test</span>
            <span style="font-size:.68rem;font-weight:700;color:#39c5cf">{s.get('rec_fix_time','2 – 4 h')}</span>
          </div>
        </div>

        <div style="font-size:.61rem;color:#c9d1d9;line-height:1.45;margin-bottom:8px;
             background:#21262d;border-radius:4px;padding:5px 7px">
          {hyp[:90]}{'…' if len(hyp)>90 else ''}
        </div>

        <button onclick="scrollToIntelligence('{sid}')"
          style="width:100%;padding:6px 0;background:{color};border:none;
          border-radius:5px;color:#fff;font-family:'IBM Plex Mono',monospace;
          font-size:.67rem;cursor:pointer;font-weight:700;letter-spacing:.04em;
          box-shadow:0 2px 8px {color}44;transition:opacity .2s"
          onmouseover="this.style.opacity='.82'"
          onmouseout="this.style.opacity='1'">
          ▶ VIEW STATION INTELLIGENCE
        </button>
      </div>
    </div>`,{{maxWidth:290,className:"orchpop"}});
  m.on("click",function(){{selectStation(sid);}});
  allM[sid]={{m:m,urgency:"{urg}",color:"{color}"}};
}})();""")

    nc = sum(1 for s in stations_data if s["urgency"]=="Critical")
    nw = sum(1 for s in stations_data if s["urgency"]=="Warning")
    nm = sum(1 for s in stations_data if s["urgency"]=="Monitor")
    mean_rul = sum(s["rul"] for s in stations_data)/max(len(stations_data),1)

    sorted_s = sorted(stations_data, key=lambda x:(
        0 if x["urgency"]=="Critical" else 1 if x["urgency"]=="Warning" else 2, x["rul"]))

    roster_html = ""
    for s in sorted_s:
        c  = urgency_color(s["urgency"])
        bg = "#1c2333" if s["id"]==selected_id else "transparent"
        bd = "#39c5cf" if s["id"]==selected_id else "transparent"
        roster_html += f"""
<div onclick="selectStation('{s['id']}')"
     style="display:flex;align-items:center;gap:7px;padding:5px 8px;cursor:pointer;
            border-radius:5px;margin-bottom:3px;background:{bg};border:1px solid {bd};
            transition:background .12s"
     onmouseover="this.style.background='#1c2333'" onmouseout="this.style.background='{bg}'">
  <div style="width:9px;height:9px;border-radius:50%;background:{c};flex-shrink:0;
    {'animation:pc 1.1s infinite' if s['urgency']=='Critical' else ''}"></div>
  <div style="flex:1;min-width:0">
    <div style="font-size:.68rem;font-weight:700;color:#e6edf3;font-family:monospace;white-space:nowrap">{s['id']}</div>
    <div style="font-size:.57rem;color:#7d8590;font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{s['city']}</div>
  </div>
  <div style="text-align:right;flex-shrink:0">
    <div style="font-size:.65rem;font-weight:700;color:{c};font-family:monospace">{s['rul']:.0f}cy</div>
    <div style="font-size:.55rem;color:#7d8590">{s['urgency'][:3]}</div>
  </div>
</div>"""

    return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:#0d1117;font-family:'IBM Plex Mono',monospace;overflow:hidden;}}
#map{{position:absolute;top:0;left:0;right:220px;bottom:0;background:#0d1117;}}
.leaflet-tile{{filter:brightness(0.48) saturate(0.45) hue-rotate(190deg);}}
.orchpop .leaflet-popup-content-wrapper{{background:#161b22!important;border:1px solid #30363d!important;border-radius:8px!important;box-shadow:0 12px 40px rgba(0,0,0,.8)!important;padding:0!important;}}
.orchpop .leaflet-popup-content{{margin:0!important;}}
.orchpop .leaflet-popup-tip-container{{display:none;}}
.leaflet-popup-close-button{{color:#7d8590!important;font-size:18px!important;top:5px!important;right:6px!important;}}
@keyframes pc{{0%,100%{{box-shadow:0 0 0 0 #ff6b3566;transform:scale(1);}}50%{{box-shadow:0 0 0 11px #ff6b3500;transform:scale(1.15);}}}}
@keyframes pw{{0%,100%{{box-shadow:0 0 0 0 #f0b42955;}}50%{{box-shadow:0 0 0 8px #f0b42900;}}}}
@keyframes pm{{0%,100%{{box-shadow:0 0 0 0 #3fb95033;}}50%{{box-shadow:0 0 0 6px #3fb95000;}}}}
.pulse-critical{{animation:pc 1.0s ease-in-out infinite;}}
.pulse-warning{{animation:pw 2.2s ease-in-out infinite;}}
.pulse-monitor{{animation:pm 3.5s ease-in-out infinite;}}
.smark-sel{{z-index:1000!important;box-shadow:0 0 0 4px rgba(255,255,255,0.3)!important;}}
/* Right panel */
#panel{{position:absolute;top:0;right:0;width:220px;bottom:0;background:rgba(13,17,23,.96);
  backdrop-filter:blur(10px);border-left:1px solid #30363d;overflow:hidden;display:flex;flex-direction:column;}}
#panel-header{{padding:12px 10px 8px;border-bottom:1px solid #30363d;flex-shrink:0;}}
#panel-kpis{{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding:8px 10px;border-bottom:1px solid #30363d;flex-shrink:0;}}
.pkpi{{background:#1c2333;border-radius:5px;padding:5px 7px;}}
.pkpi-l{{font-size:.55rem;color:#7d8590;text-transform:uppercase;letter-spacing:.06em;}}
.pkpi-v{{font-size:.92rem;font-weight:700;font-family:monospace;}}
#panel-list{{flex:1;overflow-y:auto;padding:6px 8px;}}
#panel-list::-webkit-scrollbar{{width:3px;}}
#panel-list::-webkit-scrollbar-thumb{{background:#30363d;border-radius:2px;}}
/* Legend */
#legend{{position:absolute;bottom:18px;left:14px;z-index:1000;
  background:rgba(13,17,23,.88);backdrop-filter:blur(6px);
  border:1px solid #30363d;border-radius:8px;padding:9px 13px;}}
.leg{{display:flex;align-items:center;gap:8px;font-size:.63rem;color:#c9d1d9;margin-bottom:5px;font-family:monospace;}}
.leg:last-child{{margin-bottom:0;}}
.ldot{{width:11px;height:11px;border-radius:50%;flex-shrink:0;}}
/* Leaflet controls */
.leaflet-control-attribution{{display:none;}}
.leaflet-control-zoom{{border:1px solid #30363d!important;}}
.leaflet-control-zoom a{{background:#161b22!important;color:#7d8590!important;border-color:#30363d!important;font-family:monospace!important;}}
.leaflet-control-zoom a:hover{{color:#39c5cf!important;background:#1c2333!important;}}
</style></head><body>
<div id="map"></div>
<div id="panel">
  <div id="panel-header">
    <div style="font-size:.78rem;font-weight:700;color:#39c5cf;letter-spacing:.03em">⚡ OrchestrAI</div>
    <div style="font-size:.60rem;color:#7d8590;margin-top:2px">West Africa · {len(stations_data)} Stations</div>
  </div>
  <div id="panel-kpis">
    <div class="pkpi"><div class="pkpi-l">Critical</div><div class="pkpi-v" style="color:#ff6b35">{nc}</div></div>
    <div class="pkpi"><div class="pkpi-l">Warning</div><div class="pkpi-v" style="color:#f0b429">{nw}</div></div>
    <div class="pkpi"><div class="pkpi-l">Monitor</div><div class="pkpi-v" style="color:#3fb950">{nm}</div></div>
    <div class="pkpi"><div class="pkpi-l">Mean RUL</div><div class="pkpi-v" style="color:#39c5cf">{mean_rul:.0f}</div></div>
  </div>
  <div id="panel-list">{roster_html}</div>
</div>
<div id="legend">
  <div style="font-size:.58rem;color:#7d8590;margin-bottom:6px;text-transform:uppercase;letter-spacing:.09em">Urgency Legend</div>
  <div class="leg"><div class="ldot" style="background:#ff6b35;animation:pc 1s infinite"></div>Critical · RUL ≤ 20 cy · SLA 4h</div>
  <div class="leg"><div class="ldot" style="background:#f0b429;animation:pw 2.2s infinite"></div>Warning · 21–50 cy · SLA 48h</div>
  <div class="leg"><div class="ldot" style="background:#3fb950;animation:pm 3.5s infinite"></div>Monitor · &gt;50 cy · SLA 168h</div>
</div>
<script>
var map=L.map("map",{{center:[12.0,-8.5],zoom:5,zoomControl:true,attributionControl:false}});
L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",{{maxZoom:18,subdomains:"abc"}}).addTo(map);
var allM={{}};
{"".join(markers_js)}
function selectStation(sid){{
  var e=allM[sid]; if(!e)return;
  map.flyTo(e.m.getLatLng(),9,{{duration:0.85,easeLinearity:0.4}});
  setTimeout(function(){{e.m.openPopup();}},900);
}}
function scrollToIntelligence(sid){{
  // Close popup then ask parent to scroll to + highlight the station's intelligence card
  window.parent.postMessage({{type:"scroll_to_intelligence",id:sid}},"*");
}}
window.addEventListener("message",function(e){{
  if(e.data&&e.data.type==="map_select")selectStation(e.data.id);
}});
// Auto-open popup for selected station
setTimeout(function(){{
  var sel=allM["{selected_id}"];
  if(sel){{map.setView(sel.m.getLatLng(),7);sel.m.openPopup();}}
}},400);
</script></body></html>"""
