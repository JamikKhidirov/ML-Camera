"""Веб-панель истории событий MLCamera (только стандартная библиотека).

Запуск отдельно:
    python scripts/web_panel.py --config config/config.yaml

Или вместе с камерой: в конфиге web.enabled: true, web.autostart: true
(либо флаг `python main.py --web`). Только в этом режиме доступны кнопки
управления камерами: статус, старт/стоп потока и снимок кадра.

Маршруты:
    GET  /                        — тёмная HTML-панель: история и камеры
    GET  /api/events              — JSON: события (kind, camera, q, hours, limit, offset)
    GET  /api/stats               — JSON: счётчики за период
    GET  /api/cameras             — JSON: статусы камер и доступность управления
    GET  /api/cameras/snapshot    — JPEG последнего кадра (?index=0)
    POST /api/cameras/start       — запустить камеру (?index=0)
    POST /api/cameras/stop        — остановить камеру (?index=0)
    POST /api/cameras/start_all   — запустить все камеры
    POST /api/cameras/stop_all    — остановить все камеры
    GET  /media?type=&id=         — фото события (type: face|car|animal|package)
    GET  /health                  — проверка, что сервер жив

Безопасность:
  - файлы отдаются только по id события и только из data/events/;
  - произвольные пути и обход каталога невозможны;
  - если задан web.token — нужен ?token=... (ставится cookie) или X-Token;
  - POST требует заголовок X-MLCamera и совпадающий Origin (защита от CSRF).
"""
from __future__ import annotations

import html
import json
import logging
import mimetypes
import secrets
import shutil
import sys
import threading
import time
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlsplit

from src import __version__
from src.camera_status import read_status
from src.config import Config
from src.database import Database
from src.settings import (SettingsError, apply_live, reload_cameras_async,
                          save_settings, settings_payload)

log = logging.getLogger(__name__)

_ALLOWED_TYPES = {"face", "car", "animal", "package"}

_HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#05070f; --bg2:#0a1020; --panel:rgba(17,25,43,.78); --panel2:rgba(22,32,54,.9);
  --line:rgba(93,120,168,.24); --line2:rgba(120,150,205,.4);
  --text:#eef4ff; --muted:#93a4c8; --dim:#5f7299;
  --cyan:#3ddcff; --blue:#4f8cff; --violet:#9b6bff; --green:#3ef0a6;
  --amber:#ffc857; --red:#ff5f72; --pink:#ff7bb0;
  --glow-c:rgba(61,220,255,.35); --glow-v:rgba(155,107,255,.3);
  --r:18px; --r-sm:12px;
  --shadow:0 24px 60px rgba(0,0,0,.45);
  --font:"Segoe UI",system-ui,-apple-system,Roboto,Arial,sans-serif;
  --mono:"Cascadia Mono",ui-monospace,Consolas,monospace;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{
  background:var(--bg); color:var(--text); font-family:var(--font);
  min-height:100vh; overflow-x:hidden;
}
body::before{
  content:""; position:fixed; inset:0; z-index:-3;
  background:
    radial-gradient(1100px 600px at 8% -12%, rgba(61,220,255,.14), transparent 62%),
    radial-gradient(900px 560px at 102% -4%, rgba(155,107,255,.14), transparent 60%),
    radial-gradient(700px 500px at 50% 110%, rgba(62,240,166,.07), transparent 65%),
    linear-gradient(180deg,#05070f,#080d1a 55%,#060a14);
}
body::after{
  content:""; position:fixed; inset:0; z-index:-2; pointer-events:none; opacity:.5;
  background-image:
    linear-gradient(rgba(120,160,220,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(120,160,220,.045) 1px, transparent 1px);
  background-size:54px 54px;
  mask-image:radial-gradient(circle at 50% 0%, #000 0%, transparent 78%);
  -webkit-mask-image:radial-gradient(circle at 50% 0%, #000 0%, transparent 78%);
}
.orb{position:fixed;border-radius:50%;filter:blur(90px);z-index:-1;pointer-events:none;opacity:.5}
.orb.a{width:420px;height:420px;left:-140px;top:-120px;background:rgba(61,220,255,.22);animation:drift 22s ease-in-out infinite}
.orb.b{width:380px;height:380px;right:-120px;top:180px;background:rgba(155,107,255,.2);animation:drift 27s ease-in-out infinite reverse}
@keyframes drift{50%{transform:translate3d(50px,40px,0) scale(1.12)}}
::selection{background:rgba(61,220,255,.3)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#243251;border-radius:999px;border:2px solid transparent;background-clip:padding-box}
::-webkit-scrollbar-thumb:hover{background:#33456b;background-clip:padding-box}

a{color:var(--cyan);text-decoration:none}
.wrap{max-width:1380px;margin:0 auto;padding:22px 22px 70px}

/* ---------- header ---------- */
header{
  position:sticky; top:0; z-index:30;
  background:rgba(5,7,15,.72); backdrop-filter:blur(18px) saturate(1.3);
  border-bottom:1px solid var(--line);
}
header::after{content:"";position:absolute;left:0;right:0;bottom:-1px;height:1px;
  background:linear-gradient(90deg,transparent,rgba(61,220,255,.55),rgba(155,107,255,.45),transparent)}
.head-in{max-width:1380px;margin:0 auto;padding:13px 22px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.logo{width:46px;height:46px;border-radius:15px;display:grid;place-items:center;font-size:22px;
  background:radial-gradient(circle at 30% 25%,#1b3a5c,#0c1424 70%);
  border:1px solid #2e4a75; box-shadow:0 0 0 4px rgba(61,220,255,.05), inset 0 0 18px rgba(61,220,255,.12);
  position:relative}
.logo::after{content:"";position:absolute;inset:-2px;border-radius:17px;
  background:conic-gradient(from 0deg,var(--cyan),var(--violet),var(--green),var(--cyan));
  z-index:-1;filter:blur(7px);opacity:.45;animation:spin 9s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.brand h1{font-size:19px;margin:0;letter-spacing:.2px;font-weight:800}
.brand p{margin:3px 0 0;color:var(--muted);font-size:12px;display:flex;align-items:center;gap:8px}
.pill{font-size:10px;font-weight:800;letter-spacing:.6px;text-transform:uppercase;
  padding:2px 8px;border-radius:999px;border:1px solid rgba(61,220,255,.35);
  color:var(--cyan);background:rgba(61,220,255,.08)}
#sysVersion{text-transform:none;letter-spacing:.2px}
.head-right{margin-left:auto;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.conn{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--muted);
  padding:7px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(17,25,43,.6)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 4px rgba(62,240,166,.13),0 0 12px rgba(62,240,166,.7);animation:pulse 2.2s infinite}
.conn.off .dot{background:var(--red);box-shadow:0 0 0 4px rgba(255,95,114,.14);animation:none}
.conn.off{color:#ffb7c1;border-color:rgba(255,95,114,.4)}
@keyframes pulse{50%{opacity:.35}}
.clock{font-family:var(--mono);font-size:13px;color:var(--text);letter-spacing:.5px;
  padding:7px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(17,25,43,.6);
  font-variant-numeric:tabular-nums}
.switch{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px;cursor:pointer;
  user-select:none;padding:7px 12px;border-radius:999px;border:1px solid var(--line);
  background:rgba(17,25,43,.6)}
.switch input{appearance:none;width:34px;height:19px;border-radius:999px;background:#1c2740;
  border:1px solid #31456a;position:relative;cursor:pointer;transition:.2s;margin:0}
.switch input::after{content:"";position:absolute;top:2px;left:2px;width:13px;height:13px;border-radius:50%;
  background:#7187ad;transition:.2s}
.switch input:checked{background:rgba(61,220,255,.22);border-color:rgba(61,220,255,.55)}
.switch input:checked::after{left:17px;background:var(--cyan);box-shadow:0 0 10px var(--glow-c)}

/* ---------- tabs ---------- */
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 20px;padding:6px;
  background:rgba(15,22,38,.7);border:1px solid var(--line);border-radius:16px;
  backdrop-filter:blur(10px);width:max-content;max-width:100%}
.tab{background:transparent;border:1px solid transparent;color:var(--muted);border-radius:12px;
  padding:10px 18px;font-size:13.5px;font-weight:700;cursor:pointer;transition:.18s;
  display:flex;align-items:center;gap:8px;font-family:inherit}
.tab:hover{color:var(--text);background:rgba(120,150,205,.08)}
.tab.active{color:#fff;border-color:rgba(61,220,255,.35);
  background:linear-gradient(135deg,rgba(61,220,255,.16),rgba(155,107,255,.16));
  box-shadow:0 6px 22px rgba(61,220,255,.12), inset 0 0 22px rgba(61,220,255,.07)}
.tab .n{display:inline-flex;align-items:center;justify-content:center;min-width:20px;height:20px;
  padding:0 6px;border-radius:999px;background:rgba(61,220,255,.14);border:1px solid rgba(61,220,255,.35);
  font-size:10.5px;color:var(--cyan);font-weight:800}
.tab .n:empty{display:none}

/* ---------- hero ---------- */
.hero{position:relative;overflow:hidden;border-radius:22px;border:1px solid var(--line);
  background:linear-gradient(150deg,rgba(22,33,58,.92),rgba(12,18,32,.96));
  padding:24px 26px;margin-bottom:18px;display:flex;gap:26px;align-items:center;flex-wrap:wrap;
  box-shadow:var(--shadow)}
.hero::before{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(600px 200px at 12% 0%,rgba(61,220,255,.16),transparent 70%),
             radial-gradient(500px 220px at 88% 110%,rgba(155,107,255,.16),transparent 70%)}
.hero::after{content:"";position:absolute;left:0;right:0;top:0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(61,220,255,.55),rgba(155,107,255,.5),transparent)}
.hero-copy{position:relative;z-index:1;flex:1 1 320px;min-width:260px}
.hero h2{margin:0 0 6px;font-size:clamp(20px,2.4vw,28px);letter-spacing:-.3px}
.hero p{margin:0 0 16px;color:var(--muted);font-size:13.5px;max-width:560px;line-height:1.5}
.hero-actions{display:flex;gap:9px;flex-wrap:wrap}
.ai-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:11px}
.ai-chip{font-size:10.5px;font-weight:700;color:#b9c9e8;background:rgba(9,14,26,.6);
  border:1px solid rgba(93,120,168,.28);border-radius:999px;padding:3px 9px;
  display:inline-flex;align-items:center;gap:5px}
.ai-chip b{color:var(--cyan);font-weight:800}
.mini-stats{position:relative;z-index:1;display:grid;grid-template-columns:repeat(2,minmax(128px,1fr));gap:10px;flex:1 1 330px}
.mini{background:rgba(9,14,26,.72);border:1px solid var(--line);border-radius:14px;padding:12px 14px;
  position:relative;overflow:hidden;min-width:0}
.mini::before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--cyan);opacity:.8}
.mini.v::before{background:var(--violet)} .mini.g::before{background:var(--green)}
.mini.a::before{background:var(--amber)}
.mini .mk{color:var(--dim);font-size:10.5px;font-weight:700;letter-spacing:.7px;text-transform:uppercase;display:block;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mini .mv{font-size:23px;font-weight:800;font-variant-numeric:tabular-nums;margin-top:3px;
  display:flex;align-items:baseline;gap:5px;line-height:1.1}
.mini .mv small{font-size:12px;color:var(--muted);font-weight:600}

/* ---------- buttons ---------- */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;
  background:rgba(24,35,58,.9);border:1px solid #2d3f63;color:var(--text);
  border-radius:11px;padding:10px 15px;font-size:13px;font-weight:600;cursor:pointer;
  transition:.16s;font-family:inherit;white-space:nowrap}
.btn:hover:not(:disabled){background:#213152;border-color:#40598a;transform:translateY(-1px)}
.btn:active:not(:disabled){transform:translateY(0)}
.btn:disabled{opacity:.4;cursor:not-allowed}
.btn.primary{background:linear-gradient(135deg,#0f6f91,#2459a3);border-color:#3379a8;
  box-shadow:0 6px 20px rgba(61,220,255,.16)}
.btn.primary:hover:not(:disabled){filter:brightness(1.14);
  box-shadow:0 8px 26px rgba(61,220,255,.26)}
.btn.danger{background:rgba(58,22,32,.85);border-color:#71303f;color:#ffd4db}
.btn.danger:hover:not(:disabled){background:#4a2030;border-color:#a04a5f}
.btn.ghost{background:rgba(17,25,43,.55)}
.btn.sm{padding:8px 12px;font-size:12.5px;border-radius:10px}
.btn.icon{padding:8px 11px}

/* ---------- stats / history ---------- */
.stats{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:0 0 18px}
.stat{background:linear-gradient(160deg,rgba(22,32,54,.9),rgba(12,18,32,.95));
  border:1px solid var(--line);border-radius:var(--r);padding:15px 17px;position:relative;
  overflow:hidden;transition:.18s}
.stat:hover{transform:translateY(-2px);border-color:var(--line2);box-shadow:0 14px 34px rgba(0,0,0,.32)}
.stat::after{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--cyan)}
.stat.face::after{background:var(--violet)} .stat.car::after{background:var(--green)}
.stat.animal::after{background:var(--amber)} .stat.package::after{background:var(--red)}
.stat .k{color:var(--muted);font-size:11.5px;display:flex;gap:7px;align-items:center;font-weight:600}
.stat .v{font-size:28px;font-weight:800;margin-top:6px;font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.stat .s{color:var(--dim);font-size:11px;margin-top:2px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:0 0 20px}
.controls .grow{flex:1 1 240px}
input[type=search],select{
  width:100%;background:rgba(13,20,35,.9);color:var(--text);border:1px solid var(--line);
  border-radius:11px;padding:11px 13px;font-size:13px;outline:none;font-family:inherit;
  transition:.16s;appearance:none}
input[type=search]::placeholder{color:var(--dim)}
input[type=search]:focus,select:focus{border-color:rgba(61,220,255,.55);
  box-shadow:0 0 0 3px rgba(61,220,255,.1)}
select{background-image:linear-gradient(45deg,transparent 50%,#7d90b5 50%),linear-gradient(135deg,#7d90b5 50%,transparent 50%);
  background-position:calc(100% - 17px) 50%,calc(100% - 12px) 50%;background-size:5px 5px,5px 5px;
  background-repeat:no-repeat;padding-right:34px}
.hint{color:var(--dim);font-size:12px;line-height:1.45}
.divider{display:flex;align-items:center;gap:12px;margin:24px 0 14px;color:var(--muted);
  font-size:11.5px;font-weight:800;letter-spacing:1px;text-transform:uppercase}
.divider::after{content:"";height:1px;flex:1;background:linear-gradient(90deg,var(--line),transparent)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:15px}
.card{background:linear-gradient(170deg,rgba(20,29,49,.94),rgba(11,17,30,.97));
  border:1px solid var(--line);border-radius:var(--r);overflow:hidden;cursor:pointer;
  transition:.2s;display:flex;flex-direction:column;min-height:255px;position:relative}
.card::after{content:"";position:absolute;inset:0;border-radius:var(--r);pointer-events:none;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}
.card:hover{transform:translateY(-3px);border-color:rgba(61,220,255,.4);box-shadow:0 18px 44px rgba(0,0,0,.45)}
.thumb{aspect-ratio:16/10;background:#070b14;position:relative;overflow:hidden}
.thumb img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .4s}
.card:hover .thumb img{transform:scale(1.04)}
.thumb .ph{position:absolute;inset:0;display:grid;place-items:center;font-size:38px;color:#2b3d61;
  background:repeating-linear-gradient(45deg,#0a101c,#0a101c 12px,#0c1424 12px,#0c1424 24px)}
.badge{position:absolute;top:10px;left:10px;padding:4px 10px;border-radius:999px;
  font-size:10.5px;font-weight:800;letter-spacing:.4px;text-transform:uppercase;
  backdrop-filter:blur(10px);background:rgba(5,8,16,.75);border:1px solid rgba(255,255,255,.14)}
.badge.face{color:#d0c0ff;border-color:rgba(155,107,255,.55);background:rgba(155,107,255,.14)}
.badge.car{color:#a2f8d2;border-color:rgba(62,240,166,.5);background:rgba(62,240,166,.12)}
.badge.animal{color:#ffe3a8;border-color:rgba(255,200,87,.5);background:rgba(255,200,87,.12)}
.badge.package{color:#ffbfca;border-color:rgba(255,95,114,.5);background:rgba(255,95,114,.12)}
.cbody{padding:13px 15px 14px;display:flex;flex-direction:column;gap:5px;flex:1}
.ctitle{font-size:15px;font-weight:700;line-height:1.25;word-break:break-word}
.csub{color:var(--muted);font-size:12.5px;line-height:1.4;word-break:break-word}
.cfoot{margin-top:auto;padding-top:10px;border-top:1px solid rgba(93,120,168,.18);display:flex;
  justify-content:space-between;gap:8px;color:var(--dim);font-size:11.5px}
.cfoot .cam{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:66%}
.empty{text-align:center;padding:64px 20px;color:var(--muted)}
.empty .big{font-size:46px;margin-bottom:12px;opacity:.75}
.more{display:flex;justify-content:center;margin-top:24px}
.foot{margin-top:38px;text-align:center;color:var(--dim);font-size:12px;line-height:1.6}
.foot code{color:var(--muted);background:rgba(120,150,205,.08);padding:1px 6px;border-radius:6px;font-family:var(--mono);font-size:11.5px}

/* ---------- chart ---------- */
.chart-card{background:linear-gradient(165deg,rgba(20,29,49,.92),rgba(10,15,27,.96));
  border:1px solid var(--line);border-radius:20px;padding:18px 20px 12px;margin-bottom:18px;
  box-shadow:var(--shadow);position:relative;overflow:hidden}
.chart-card::before{content:"";position:absolute;inset:0 0 auto 0;height:1px;
  background:linear-gradient(90deg,transparent,rgba(61,220,255,.4),transparent)}
.chart-head{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:12px}
.chart-head h3{margin:0;font-size:15px}
.chart-head .hint{margin-left:auto}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11.5px;align-items:center}
.legend i{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.legend .f i{background:var(--violet)} .legend .c i{background:var(--green)}
.legend .a i{background:var(--amber)} .legend .p i{background:var(--red)}
.chart{display:flex;align-items:flex-end;gap:3px;height:150px;padding-top:6px}
.bar-col{flex:1;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;
  height:100%;position:relative;gap:5px;min-width:0;cursor:default}
.bar-stack{width:100%;max-width:26px;display:flex;flex-direction:column-reverse;border-radius:5px 5px 2px 2px;
  overflow:hidden;min-height:2px;background:rgba(120,150,205,.09);
  transition:height .5s cubic-bezier(.22,1,.36,1)}
.bar-stack:hover{filter:brightness(1.25);box-shadow:0 0 14px rgba(61,220,255,.25)}
.seg{width:100%;min-height:0}
.seg.face{background:linear-gradient(180deg,#b18cff,#7b4ffe)}
.seg.car{background:linear-gradient(180deg,#6cf7c0,#25c98a)}
.seg.animal{background:linear-gradient(180deg,#ffdb8a,#e8a92c)}
.seg.package{background:linear-gradient(180deg,#ff98a9,#e0405a)}
.bar-lbl{font-size:9.5px;color:var(--dim);white-space:nowrap;overflow:hidden;max-width:100%;
  font-variant-numeric:tabular-nums}
.chart-foot{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;
  color:var(--dim);font-size:11px;border-top:1px solid rgba(93,120,168,.14);padding-top:9px;margin-top:8px}

/* ---------- cameras ---------- */
.note{color:var(--muted);font-size:12.5px;line-height:1.5;background:rgba(20,30,50,.75);
  border:1px solid var(--line);border-left:3px solid var(--amber);border-radius:12px;
  padding:11px 14px;margin:0 0 16px}
.cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:16px}
.cam{background:linear-gradient(170deg,rgba(20,29,49,.94),rgba(10,16,28,.98));
  border:1px solid var(--line);border-radius:20px;overflow:hidden;display:flex;flex-direction:column;
  transition:.2s;position:relative}
.cam:hover{border-color:var(--line2);box-shadow:0 20px 50px rgba(0,0,0,.42)}
.cam.is-live{border-color:rgba(62,240,166,.28);
  box-shadow:0 0 0 1px rgba(62,240,166,.08),0 18px 44px rgba(0,0,0,.4)}
.cam.is-live::after{content:"";position:absolute;left:0;right:0;top:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--green),transparent);opacity:.85}
.cam-live{position:relative;aspect-ratio:16/9;background:#04070d;overflow:hidden;cursor:pointer}
.cam-live img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;
  transition:opacity .35s;display:block}
.cam-live img.on{opacity:1}
.cam-ph{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;
  gap:6px;color:#31456c;text-align:center;padding:16px;
  background:
    radial-gradient(400px 200px at 50% 40%,rgba(61,220,255,.05),transparent 70%),
    repeating-linear-gradient(45deg,#080d18,#080d18 14px,#0a1120 14px,#0a1120 28px)}
.cam-ph .pi{font-size:34px;filter:grayscale(1);opacity:.8}
.cam-ph .pt{font-size:12.5px;font-weight:700;color:#5a7099}
.cam-ph .ps{font-size:11px;color:#3d5279;max-width:240px;line-height:1.4}
.live-top{position:absolute;top:10px;left:10px;right:10px;display:flex;align-items:center;gap:8px;z-index:2}
.live-badge{margin-left:auto;display:none;align-items:center;gap:5px;font-size:10px;font-weight:900;
  letter-spacing:1px;color:#ffdfe4;background:rgba(255,95,114,.2);border:1px solid rgba(255,95,114,.55);
  padding:3px 8px;border-radius:999px;backdrop-filter:blur(8px)}
.live-badge::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--red);
  box-shadow:0 0 8px var(--red);animation:blink 1.4s infinite}
@keyframes blink{50%{opacity:.25}}
.cam.is-live .live-badge{display:inline-flex}
.live-bot{position:absolute;left:10px;right:10px;bottom:10px;display:flex;gap:5px;flex-wrap:wrap;z-index:2}
.lv-chip{font-size:10.5px;font-weight:700;color:#c6d6f2;background:rgba(4,8,16,.68);
  border:1px solid rgba(140,170,220,.24);border-radius:7px;padding:3px 7px;
  backdrop-filter:blur(8px);font-variant-numeric:tabular-nums}
.live-hover{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  background:rgba(4,8,16,.45);backdrop-filter:blur(2px);color:#eaf3ff;font-size:12.5px;font-weight:700;
  opacity:0;transition:.2s;z-index:1;gap:7px}
.cam-live:hover .live-hover{opacity:1}
.scan{position:absolute;left:0;right:0;height:22%;top:-25%;z-index:1;pointer-events:none;display:none;
  background:linear-gradient(180deg,transparent,rgba(61,220,255,.14),transparent);
  animation:scan 4.5s linear infinite}
.cam.is-live .scan{display:block}
@keyframes scan{0%{top:-25%}100%{top:105%}}
.cam-body{padding:14px 16px 15px;display:flex;flex-direction:column;gap:11px;flex:1}
.cam-head{display:flex;align-items:center;gap:9px}
.cam-name{font-size:15.5px;font-weight:800;line-height:1.25;word-break:break-word}
.cam-idx{margin-left:auto;font-family:var(--mono);font-size:10.5px;color:var(--dim);
  border:1px solid var(--line);border-radius:6px;padding:2px 6px}
.cam-state{flex:0 0 auto;font-size:10.5px;font-weight:800;padding:4px 9px;border-radius:999px;
  border:1px solid #33456a;color:var(--muted);background:rgba(19,28,48,.8);white-space:nowrap;
  display:inline-flex;align-items:center;gap:5px;text-transform:lowercase}
.cam-state::before{content:"";width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.85}
.st-running{color:#8cf7cd;border-color:rgba(62,240,166,.45);background:rgba(62,240,166,.1)}
.st-starting,.st-stopping{color:#ffe0a3;border-color:rgba(255,200,87,.45);background:rgba(255,200,87,.1)}
.st-error{color:#ffb3bd;border-color:rgba(255,95,114,.45);background:rgba(255,95,114,.1)}
.st-stopped{color:#a7b8d6}
.cam-meta{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px}
.cm{background:rgba(9,14,25,.6);border:1px solid rgba(93,120,168,.16);border-radius:9px;
  padding:7px 8px;min-width:0}
.cm .ck{display:block;color:var(--dim);font-size:9px;font-weight:800;letter-spacing:.5px;
  text-transform:uppercase;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cm .cv{display:block;font-size:12.5px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cam-extra{color:var(--dim);font-size:11px;line-height:1.4;min-height:0}
.cam-extra:empty{display:none}
.cam-err{color:#ffb3bd;font-size:11.5px;line-height:1.4;word-break:break-word;
  background:rgba(255,95,114,.08);border:1px solid rgba(255,95,114,.28);
  border-radius:9px;padding:7px 9px}
.cam-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:auto}
.cam-actions .btn{flex:1 1 auto}
.cam-actions .btn.icon{flex:0 0 auto}

/* ---------- system ---------- */
.sys-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:18px}
.sys-card{background:linear-gradient(160deg,rgba(21,31,52,.92),rgba(10,16,28,.96));
  border:1px solid var(--line);border-radius:16px;padding:15px 17px;position:relative;overflow:hidden}
.sys-card::before{content:"";position:absolute;inset:0 auto 0 0;width:3px;background:var(--cyan);opacity:.8}
.sys-card.v::before{background:var(--violet)} .sys-card.g::before{background:var(--green)}
.sys-card.a::before{background:var(--amber)}
.sys-card .sk{color:var(--dim);font-size:10.5px;font-weight:800;letter-spacing:.7px;text-transform:uppercase}
.sys-card .sv{font-size:22px;font-weight:800;margin-top:5px;font-variant-numeric:tabular-nums;
  letter-spacing:-.3px;word-break:break-word}
.sys-card .ss{color:var(--muted);font-size:11.5px;margin-top:3px;line-height:1.4}
.bar{height:7px;border-radius:999px;background:rgba(120,150,205,.14);overflow:hidden;margin-top:9px}
.bar span{display:block;height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width .6s}
.chips{display:flex;gap:8px;flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:700;
  padding:6px 11px;border-radius:999px;border:1px solid var(--line);color:var(--muted);
  background:rgba(17,25,43,.6)}
.chip::before{content:"";width:7px;height:7px;border-radius:50%;background:#3a4d70}
.chip.on{color:#a5f8d6;border-color:rgba(62,240,166,.4);background:rgba(62,240,166,.08)}
.chip.on::before{background:var(--green);box-shadow:0 0 8px rgba(62,240,166,.8)}
.chip.off{opacity:.6}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:16px;background:rgba(12,19,33,.7)}
table{width:100%;border-collapse:collapse;font-size:12.5px;min-width:720px}
th,td{padding:10px 13px;text-align:left;border-bottom:1px solid rgba(93,120,168,.14);white-space:nowrap}
th{color:var(--dim);font-size:10.5px;letter-spacing:.7px;text-transform:uppercase;font-weight:800;
  background:rgba(17,25,43,.6);position:sticky;top:0}
tbody tr:hover{background:rgba(120,150,205,.05)}
tbody tr:last-child td{border-bottom:0}
.sys-block{margin-bottom:18px}
.sys-block h3{font-size:14px;margin:0 0 10px;display:flex;align-items:center;gap:8px}
.sys-block h3 .hint{font-weight:400;margin-left:auto}
.endpoints{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:8px}
.ep{display:flex;gap:9px;align-items:center;background:rgba(12,19,33,.7);border:1px solid var(--line);
  border-radius:10px;padding:8px 11px;font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.ep b{color:var(--cyan);font-weight:700;flex:0 0 auto}
.ep span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ---------- modal / toast ---------- */
.modal{position:fixed;inset:0;z-index:60;display:none;align-items:center;justify-content:center;
  padding:22px;background:rgba(3,6,12,.84);backdrop-filter:blur(10px)}
.modal.open{display:flex;animation:fade .18s ease-out}
@keyframes fade{from{opacity:0}to{opacity:1}}
.modal-in{background:linear-gradient(170deg,#0f1728,#0a0f1b);border:1px solid #2c3c5e;border-radius:20px;
  max-width:1060px;width:100%;max-height:92vh;overflow:auto;box-shadow:0 40px 100px rgba(0,0,0,.7);
  animation:pop .2s cubic-bezier(.2,1.3,.4,1)}
@keyframes pop{from{transform:translateY(14px) scale(.985);opacity:.5}to{transform:none;opacity:1}}
.modal-head{display:flex;align-items:center;gap:12px;padding:15px 19px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:rgba(15,23,40,.96);backdrop-filter:blur(10px);z-index:3}
.modal-head h3{margin:0;font-size:16.5px;letter-spacing:-.2px}
.modal-head .badge{position:static;flex:0 0 auto}
.modal-head .x{margin-left:auto;background:rgba(120,150,205,.08);border:1px solid var(--line);
  color:var(--muted);font-size:22px;cursor:pointer;line-height:1;padding:2px 9px 5px;border-radius:9px;
  transition:.16s;font-family:inherit}
.modal-head .x:hover{color:var(--text);background:rgba(255,95,114,.14);border-color:rgba(255,95,114,.4)}
.modal-body{padding:18px;display:flex;flex-direction:column;gap:16px}
.modal-body img{width:100%;border-radius:12px;background:#05080f;display:block}
.modal-body .duo{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.modal-body .duo img{height:100%;object-fit:contain}
.cam-modal{display:grid;grid-template-columns:1.45fr 1fr;gap:16px;align-items:start}
.cam-view{position:relative;aspect-ratio:16/9;border-radius:14px;overflow:hidden;background:#04070d;
  border:1px solid var(--line)}
.cam-view img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;border-radius:0}
.cam-view .cam-ph{border-radius:0}
.kv{display:grid;grid-template-columns:auto 1fr;gap:7px 14px;font-size:12.5px}
.kv dt{color:var(--dim)} .kv dd{margin:0;color:var(--text);font-weight:600;text-align:right;
  font-variant-numeric:tabular-nums;word-break:break-word}
.meta{display:flex;flex-wrap:wrap;gap:8px 20px;color:var(--muted);font-size:13px}
.meta b{color:var(--text)}
.mhead{display:flex;align-items:center;gap:10px;margin-bottom:8px}
.mhead h4{margin:0;font-size:13.5px}
.mhead .btn{margin-left:auto}
.mini-events{display:flex;flex-direction:column;gap:7px}
.mev{display:flex;gap:10px;align-items:center;background:rgba(13,20,35,.7);border:1px solid var(--line);
  border-radius:11px;padding:7px 9px;cursor:pointer;transition:.15s;text-align:left;font-family:inherit;
  color:var(--text);width:100%}
.mev:hover{background:rgba(30,45,72,.7);border-color:var(--line2)}
.mev img{width:46px;height:34px;object-fit:cover;border-radius:7px;flex:0 0 auto;background:#080d18}
.mev .mt{min-width:0;flex:1}
.mev .mt b{display:block;font-size:12.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mev .mt span{display:block;font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mev .ma{font-size:10.5px;color:var(--dim);flex:0 0 auto}
.toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(100px);
  background:linear-gradient(150deg,#1b2946,#121b2e);border:1px solid #3a4f78;color:var(--text);
  padding:11px 18px;border-radius:13px;font-size:13px;z-index:90;opacity:0;transition:.28s;
  box-shadow:0 18px 50px rgba(0,0,0,.5);display:flex;align-items:center;gap:9px;max-width:calc(100vw - 32px)}
.toast.show{transform:translateX(-50%) translateY(0);opacity:1}
.toast.err{border-color:rgba(255,95,114,.55);background:linear-gradient(150deg,#3a1822,#241019)}
.toast.ok{border-color:rgba(62,240,166,.45)}
.toast .ti{font-size:15px}

/* ---------- настройки ---------- */
.set-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;
  flex-wrap:wrap;background:linear-gradient(165deg,rgba(20,30,50,.9),rgba(9,14,25,.96));
  border:1px solid var(--line);border-radius:20px;padding:18px 20px;margin-bottom:16px}
.set-head h2{margin:0 0 5px;font-size:19px}
.set-head p{margin:0;color:var(--muted);font-size:12.5px;line-height:1.5;max-width:640px}
.set-head code{color:var(--cyan);font-family:var(--mono);font-size:11.5px}
.set-actions{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.check-inline{display:inline-flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);
  background:rgba(9,14,26,.55);border:1px solid var(--line);border-radius:10px;padding:8px 11px;cursor:pointer}
.check-inline input{accent-color:var(--cyan);width:15px;height:15px;margin:0}
.set-card{background:linear-gradient(170deg,rgba(18,27,46,.92),rgba(9,14,25,.97));
  border:1px solid var(--line);border-radius:18px;padding:16px 18px;margin-bottom:14px}
.set-card>h3{margin:0 0 4px;font-size:14.5px;display:flex;align-items:center;gap:8px}
.set-card>p{margin:0 0 13px;color:var(--dim);font-size:11.5px;line-height:1.45}
.set-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(235px,1fr));gap:12px}
.set-field{display:flex;flex-direction:column;gap:5px;min-width:0}
.set-field>label{font-size:11.5px;color:var(--muted);font-weight:600;display:flex;
  justify-content:space-between;gap:8px;align-items:baseline}
.set-field>label .unit{font-size:10px;color:var(--dim);font-weight:500}
.set-field input[type=text],.set-field input[type=number],.set-field input[type=password],
.set-field input[type=time],.set-field select{
  width:100%;background:rgba(6,10,19,.85);border:1px solid var(--line);border-radius:9px;
  color:var(--text);padding:8px 10px;font-family:inherit;font-size:12.5px;outline:none;
  transition:border-color .15s,box-shadow .15s}
.set-field input:focus,.set-field select:focus{border-color:rgba(61,220,255,.6);
  box-shadow:0 0 0 3px rgba(61,220,255,.1)}
.set-field .fh{font-size:10.5px;color:var(--dim);line-height:1.35}
.set-field.bool{flex-direction:row;align-items:center;justify-content:space-between;gap:12px;
  background:rgba(8,13,24,.55);border:1px solid rgba(93,120,168,.18);border-radius:10px;
  padding:9px 11px;min-height:44px}
.set-field.bool>label{font-size:12px;color:var(--text);font-weight:500;cursor:pointer}
.set-field.bool input{accent-color:var(--cyan);width:17px;height:17px;flex:0 0 auto;cursor:pointer}
.set-foot{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:4px 0 10px}
.set-foot .hint{font-size:11.5px;color:var(--dim)}
.cam-edit-list{display:flex;flex-direction:column;gap:9px}
.cam-row{display:grid;grid-template-columns:1.1fr 1.7fr auto auto;gap:8px;align-items:center;
  background:rgba(8,13,24,.55);border:1px solid rgba(93,120,168,.2);border-radius:11px;padding:8px 9px}
.cam-row input[type=text]{width:100%;background:rgba(6,10,19,.85);border:1px solid var(--line);
  border-radius:8px;color:var(--text);padding:7px 9px;font-family:inherit;font-size:12px;outline:none}
.cam-row input[type=text]:focus{border-color:rgba(61,220,255,.55)}
.cam-row label{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:var(--muted);white-space:nowrap}
.cam-row input[type=checkbox]{accent-color:var(--cyan);width:15px;height:15px;margin:0}
.cam-row .rm{background:rgba(255,95,114,.1);border:1px solid rgba(255,95,114,.3);
  color:#ff8b98;border-radius:8px;width:28px;height:28px;cursor:pointer;font-size:14px;line-height:1}
.cam-row .rm:hover{background:rgba(255,95,114,.2)}
.cam-add{margin-top:10px}
@media(max-width:700px){
  .set-head{padding:15px}
  .set-actions{width:100%}
  .set-actions .btn{flex:1}
  .cam-row{grid-template-columns:1fr 1fr auto}
  .cam-row .cam-enabled{grid-column:1 / -1}
}

/* ---------- responsive ---------- */
@media(max-width:1100px){
  .sys-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .cam-modal{grid-template-columns:1fr}
}
@media(max-width:900px){
  .stats{grid-template-columns:repeat(3,minmax(0,1fr))}
  .cams{grid-template-columns:1fr}
  .cam-meta{grid-template-columns:repeat(4,minmax(0,1fr))}
}
@media(max-width:660px){
  .wrap{padding:16px 13px 50px}
  .head-in{padding:11px 13px}
  .brand h1{font-size:16.5px}
  .brand p{display:none}
  .head-right{gap:7px}
  .clock{display:none}
  .stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .hero{padding:18px 17px}
  .mini-stats{grid-template-columns:repeat(2,minmax(0,1fr));width:100%}
  .grid{grid-template-columns:1fr 1fr}
  .tabs{width:100%;justify-content:space-between}
  .tab{padding:9px 11px;font-size:12.5px;flex:1;justify-content:center}
  .sys-grid{grid-template-columns:1fr}
  .modal-body .duo{grid-template-columns:1fr}
  .chart{height:120px}
  .bar-lbl{font-size:8px}
}
@media(max-width:440px){
  .grid{grid-template-columns:1fr}
  .cam-meta{grid-template-columns:repeat(2,minmax(0,1fr))}
  .live-bot .lv-chip:nth-child(3){display:none}
}
@media(prefers-reduced-motion:reduce){
  *{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
}
</style>
</head>
<body>
<div class="orb a"></div><div class="orb b"></div>

<header>
  <div class="head-in">
    <div class="logo">🎥</div>
    <div class="brand">
      <h1>__TITLE__</h1>
      <p>локальный видеонаблюдатель <span class="pill" id="sysVersion">v1.1.0</span></p>
    </div>
    <div class="head-right">
      <div class="conn" id="conn"><span class="dot"></span><span id="statusText">онлайн</span></div>
      <div class="clock" id="clock">--:--:--</div>
      <label class="switch"><input type="checkbox" id="autoRefresh" checked><span>авто</span></label>
    </div>
  </div>
</header>

<main class="wrap">
  <nav class="tabs">
    <button class="tab active" data-tab="cameras">🎥 Камеры <span class="n" id="camBadge"></span></button>
    <button class="tab" data-tab="events">🗂 История</button>
    <button class="tab" data-tab="settings">🛠 Настройки</button>
    <button class="tab" data-tab="system">⚙️ Система</button>
  </nav>

  <!-- ================= КАМЕРЫ ================= -->
  <section id="camerasView">
    <div class="hero">
      <div class="hero-copy">
        <h2>Живой обзор</h2>
        <p>Поток с камер, рамки объектов и «чертёж лица» по ИИ — прямо в браузере.</p>
        <div class="hero-actions">
          <button class="btn primary" id="startAll">▶ Запустить все</button>
          <button class="btn danger" id="stopAll">■ Остановить все</button>
          <button class="btn ghost" id="camRefresh">⟳ Обновить</button>
        </div>
        <div class="ai-chips">
          <span class="ai-chip"><b>AI</b> лица · 106 точек</span>
          <span class="ai-chip"><b>AI</b> машины и номера</span>
          <span class="ai-chip"><b>AI</b> животные</span>
          <span class="ai-chip"><b>AI</b> оставленные вещи</span>
        </div>
      </div>
      <div class="mini-stats">
        <div class="mini g">
          <span class="mk">работают</span>
          <span class="mv"><span id="heroRun">0</span><small>/ <span id="heroTotal">0</span></small></span>
        </div>
        <div class="mini v">
          <span class="mk">событий за 24 ч</span>
          <span class="mv" id="heroEvents">0</span>
        </div>
        <div class="mini">
          <span class="mk">средний fps обработки</span>
          <span class="mv" id="heroFps">0<small>fps</small></span>
        </div>
        <div class="mini a">
          <span class="mk">кадров обработано</span>
          <span class="mv" id="heroFrames">0</span>
        </div>
      </div>
    </div>

    <div class="note" id="camNote" style="display:none"></div>
    <div class="cams" id="camList"></div>
  </section>

  <!-- ================= ИСТОРИЯ ================= -->
  <section id="eventsView" style="display:none">
    <div class="chart-card">
      <div class="chart-head">
        <h3>📈 Активность</h3>
        <span class="hint" id="chartTitle">за последние 24 часа</span>
        <div class="legend">
          <span class="f"><i></i>люди</span>
          <span class="c"><i></i>машины</span>
          <span class="a"><i></i>животные</span>
          <span class="p"><i></i>вещи</span>
        </div>
      </div>
      <div class="chart" id="chart"></div>
      <div class="chart-foot">
        <span>Наведи на столбец — увидишь разбивку по типам.</span>
        <span id="chartFoot"></span>
      </div>
    </div>

    <section class="stats">
      <div class="stat total"><div class="k">📊 Всего</div><div class="v" id="sTotal">0</div><div class="s">за выбранный период</div></div>
      <div class="stat face"><div class="k">👤 Люди</div><div class="v" id="sFaces">0</div><div class="s" id="sUnknown">незнакомцев: 0</div></div>
      <div class="stat car"><div class="k">🚗 Машины</div><div class="v" id="sCars">0</div><div class="s">с номерами и марками</div></div>
      <div class="stat animal"><div class="k">🐾 Животные</div><div class="v" id="sAnimals">0</div><div class="s">собаки, коты, птицы</div></div>
      <div class="stat package"><div class="k">📦 Вещи</div><div class="v" id="sPackages">0</div><div class="s">оставленные без присмотра</div></div>
    </section>

    <section class="controls">
      <div class="grow"><input type="search" id="search" placeholder="Поиск: имя, номер, марка, класс…"></div>
      <div><select id="type">
        <option value="all">Все события</option>
        <option value="face">Люди</option>
        <option value="car">Машины</option>
        <option value="animal">Животные</option>
        <option value="package">Посылки и вещи</option>
      </select></div>
      <div><select id="camera"><option value="">Все камеры</option></select></div>
      <div><select id="hours">
        <option value="24">За 24 часа</option>
        <option value="72">За 3 дня</option>
        <option value="168">За 7 дней</option>
        <option value="0">За всё время</option>
      </select></div>
      <button class="btn primary" id="refresh">Обновить</button>
    </section>

    <div id="list"></div>
    <div class="more" id="moreWrap" style="display:none"><button class="btn" id="more">Показать ещё</button></div>
  </section>

  <!-- ================= НАСТРОЙКИ ================= -->
  <section id="settingsView" style="display:none">
    <div class="set-head">
      <div>
        <h2>🛠 Настройки</h2>
        <p>Один конфиг для всего: <code id="setPath">config/config.yaml</code>.
          Меняй поля и нажимай «Сохранить и применить» — камеры перезапустятся сами.</p>
      </div>
      <div class="set-actions">
        <label class="check-inline"><input type="checkbox" id="setRestart" checked>
          перезапустить камеры</label>
        <button class="btn ghost" id="setReload">⟳ Вернуть</button>
        <button class="btn primary" id="setSave">💾 Сохранить и применить</button>
      </div>
    </div>
    <div class="note" id="setHint" style="display:none"></div>
    <div id="setForm"><div class="hint">Загрузка настроек…</div></div>
  </section>

  <!-- ================= СИСТЕМА ================= -->
  <section id="systemView" style="display:none">
    <div class="sys-grid">
      <div class="sys-card">
        <div class="sk">версия</div>
        <div class="sv" id="sysVersionBig">—</div>
        <div class="ss">MLCamera · локальная обработка</div>
      </div>
      <div class="sys-card v">
        <div class="sk">аптайм панели</div>
        <div class="sv" id="sysUptime">—</div>
        <div class="ss">с момента запуска веб-сервера</div>
      </div>
      <div class="sys-card g">
        <div class="sk">база событий</div>
        <div class="sv" id="sysDb">—</div>
        <div class="ss">SQLite + WAL, локально</div>
      </div>
      <div class="sys-card a">
        <div class="sk">свободно на диске</div>
        <div class="sv" id="sysDisk">—</div>
        <div class="bar"><span id="sysDiskBar" style="width:0%"></span></div>
        <div class="ss" id="sysDiskText">—</div>
      </div>
    </div>

    <div class="sys-block">
      <h3>🎛 Возможности <span class="hint" id="sysControllable"></span></h3>
      <div class="chips" id="sysFeatures"></div>
    </div>

    <div class="sys-block">
      <h3>📷 Камеры <span class="hint">живые статусы и накопленная статистика</span></h3>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>#</th><th>Камера</th><th>Статус</th><th>Источник</th>
            <th>Разрешение</th><th>Обработка</th><th>Аптайм</th><th>События 24 ч</th>
          </tr></thead>
          <tbody id="sysCamTable"><tr><td colspan="8" class="hint">Загрузка…</td></tr></tbody>
        </table>
      </div>
    </div>

    <div class="sys-block">
      <h3>🔌 API <span class="hint">всё локально, без внешних сервисов</span></h3>
      <div class="endpoints">
        <div class="ep"><b>GET</b><span>/api/events</span></div>
        <div class="ep"><b>GET</b><span>/api/stats</span></div>
        <div class="ep"><b>GET</b><span>/api/timeline</span></div>
        <div class="ep"><b>GET</b><span>/api/system</span></div>
        <div class="ep"><b>GET</b><span>/api/settings</span></div>
        <div class="ep"><b>POST</b><span>/api/settings</span></div>
        <div class="ep"><b>GET</b><span>/api/cameras</span></div>
        <div class="ep"><b>GET</b><span>/api/cameras/stream</span></div>
        <div class="ep"><b>GET</b><span>/api/cameras/snapshot</span></div>
        <div class="ep"><b>POST</b><span>/api/cameras/start|stop</span></div>
      </div>
    </div>

    <div class="foot">MLCamera · события хранятся в локальной SQLite · фото — в <code>data/events/</code></div>
  </section>

  <div class="foot" id="mainFoot">MLCamera · события хранятся в локальной SQLite · фото — в <code>data/events/</code></div>
</main>

<div class="modal" id="modal">
  <div class="modal-in">
    <div class="modal-head">
      <span id="mBadge" class="badge"></span>
      <h3 id="mTitle"></h3>
      <button class="x" id="mClose" aria-label="Закрыть">×</button>
    </div>
    <div class="modal-body" id="mBody"></div>
  </div>
</div>

<div class="toast" id="toast"><span class="ti">✓</span><span id="toastText"></span></div>

<script>
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const state = {type:"all", camera:"", q:"", hours:24, offset:0, limit:30, events:[], hasMore:false};
const TYPE_LABEL = {face:"Лицо", car:"Машина", animal:"Животное", package:"Посылка"};
const TYPE_ICON = {face:"👤", car:"🚗", animal:"🐾", package:"📦"};
const SRC_KIND = {rtsp:"RTSP", http:"HTTP", webcam:"веб-камера", file:"файл"};
const CAM_STATE = {
  running:["работает","st-running"], starting:["запуск…","st-starting"],
  stopping:["остановка…","st-stopping"], stopped:["остановлена","st-stopped"],
  error:["ошибка","st-error"]
};
let cameras = [], controllable = false, staleStatus = false;
let streamCfg = {fps: 10, quality: 70};
let autoTimer = null, autoOn = true, lastDate = null;
let activeTab = "cameras";
let modalKind = null, modalCamIndex = -1, modalTimer = null;
const camCards = new Map();

/* ---------- утилиты ---------- */
function el(tag, cls, text){
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined && text !== null) n.textContent = text;
  return n;
}
function fmtNum(n){
  n = Number(n) || 0;
  if (n >= 1000000) return (n/1000000).toFixed(1).replace(".0","") + "M";
  if (n >= 1000) return (n/1000).toFixed(1).replace(".0","") + "k";
  return String(Math.round(n));
}
function fmtBytes(n){
  n = Number(n) || 0;
  if (n >= 1024**3) return (n/1024**3).toFixed(1) + " ГБ";
  if (n >= 1024**2) return (n/1024**2).toFixed(1) + " МБ";
  if (n >= 1024) return (n/1024).toFixed(0) + " КБ";
  return n + " Б";
}
function fmtDuration(sec){
  sec = Math.max(0, Math.round(Number(sec) || 0));
  const d = Math.floor(sec/86400), h = Math.floor(sec%86400/3600),
        m = Math.floor(sec%3600/60), s = sec%60;
  if (d) return d + " д " + h + " ч";
  if (h) return h + " ч " + m + " м";
  if (m) return m + " мин " + s + " с";
  return s + " с";
}
function fmtTime(ts){
  return new Date(ts*1000).toLocaleTimeString("ru-RU", {hour:"2-digit", minute:"2-digit"});
}
function fmtDate(ts){
  return new Date(ts*1000).toLocaleDateString("ru-RU", {weekday:"long", day:"numeric", month:"long", year:"numeric"});
}
function fmtDateTime(ts){ return new Date(ts*1000).toLocaleString("ru-RU"); }
function ago(ts){
  if (!ts) return "нет кадров";
  const s = Math.max(0, Math.round(Date.now()/1000 - ts));
  if (s < 5) return "только что";
  if (s < 60) return s + " с назад";
  if (s < 3600) return Math.round(s/60) + " мин назад";
  if (s < 86400) return Math.round(s/3600) + " ч назад";
  return Math.round(s/86400) + " дн назад";
}
function mediaUrl(e, crop){
  return "/media?type=" + encodeURIComponent(e.type) + "&id=" + encodeURIComponent(e.id) + (crop ? "&crop=1" : "");
}
function toast(msg, kind){
  const t = $("#toast");
  $("#toastText").textContent = msg;
  t.className = "toast show" + (kind ? " " + kind : "");
  t.querySelector(".ti").textContent = kind === "err" ? "!" : (kind === "ok" ? "✓" : "•");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 3400);
}
function setOnline(ok, label){
  const c = $("#conn");
  c.classList.toggle("off", !ok);
  $("#statusText").textContent = label || (ok ? "онлайн" : "нет связи");
}
function liveUrl(index){
  return "/api/cameras/stream?index=" + encodeURIComponent(index) +
         "&fps=" + (streamCfg.fps || 10) +
         "&quality=" + (streamCfg.quality || 70) + "&t=" + Date.now();
}
async function api(path, params){
  const qs = new URLSearchParams();
  Object.entries(params || {}).forEach(([k,v]) => { if (v !== "" && v !== null && v !== undefined) qs.set(k,v); });
  const r = await fetch(path + (qs.toString() ? "?" + qs.toString() : ""),
                        {cache:"no-store", credentials:"same-origin"});
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}
async function apiPost(path, payload){
  const r = await fetch(path, {
    method:"POST", cache:"no-store", credentials:"same-origin",
    headers:{"Content-Type":"application/json", "X-MLCamera":"1"},
    body:JSON.stringify(payload)
  });
  let data = {};
  try { data = await r.json(); } catch(_) {}
  if (!r.ok){
    const err = new Error(data.error || data.message || ("HTTP " + r.status));
    err.data = data;
    throw err;
  }
  return data;
}

/* ---------- камеры ---------- */
async function camAction(action, index, btn){
  if (btn) btn.disabled = true;
  try{
    const url = "/api/cameras/" + action + (index === null || index === undefined ? "" : "?index=" + encodeURIComponent(index));
    const r = await fetch(url, {
      method:"POST", headers:{"X-MLCamera":"1"}, cache:"no-store", credentials:"same-origin"
    });
    let data = {};
    try { data = await r.json(); } catch(_) {}
    if (!r.ok) throw new Error(data.error || data.message || ("HTTP " + r.status));
    toast(action === "start" || action === "start_all" ? "Камера запускается…" : "Камера останавливается…", "ok");
    if (data.cameras) applyCameras(data.cameras, controllable, false);
    setTimeout(loadCameras, 700);
  }catch(err){
    toast("Не удалось: " + err.message, "err");
  }finally{
    if (btn) btn.disabled = false;
  }
}

function makeCamCard(c){
  const card = el("article", "cam");
  card.dataset.index = String(c.index);

  const liveWrap = el("div", "cam-live");
  const img = el("img"); img.alt = "Живой поток: " + (c.name || "");
  const ph = el("div", "cam-ph");
  ph.appendChild(el("div", "pi", "📷"));
  const pt = el("div", "pt", "нет сигнала");
  const ps = el("div", "ps", "нажми «Запустить»");
  ph.appendChild(pt); ph.appendChild(ps);
  const scan = el("div", "scan");
  const top = el("div", "live-top");
  const stateEl = el("span", "cam-state st-stopped", "—");
  const liveBadge = el("span", "live-badge", "LIVE");
  top.appendChild(stateEl); top.appendChild(liveBadge);
  const bot = el("div", "live-bot");
  const resEl = el("span", "lv-chip", "—");
  const fpsEl = el("span", "lv-chip", "— fps");
  const srcEl = el("span", "lv-chip", "—");
  bot.appendChild(resEl); bot.appendChild(fpsEl); bot.appendChild(srcEl);
  const hover = el("div", "live-hover", "⛶ Открыть камеру");
  liveWrap.appendChild(img); liveWrap.appendChild(ph); liveWrap.appendChild(scan);
  liveWrap.appendChild(top); liveWrap.appendChild(bot); liveWrap.appendChild(hover);
  liveWrap.addEventListener("click", () => openCameraModal(currentCam(c.index)));

  const body = el("div", "cam-body");
  const head = el("div", "cam-head");
  const nameEl = el("div", "cam-name", c.name || ("Камера " + (c.index + 1)));
  head.appendChild(nameEl);
  head.appendChild(el("span", "cam-idx", "#" + (c.index + 1)));
  body.appendChild(head);

  const meta = el("div", "cam-meta");
  const mk = (label) => {
    const box = el("div", "cm");
    box.appendChild(el("span", "ck", label));
    const v = el("span", "cv", "—");
    box.appendChild(v);
    meta.appendChild(box);
    return v;
  };
  const framesEl = mk("кадров");
  const procEl = mk("обработка");
  const uptimeEl = mk("аптайм");
  const eventsEl = mk("события 24ч");
  body.appendChild(meta);

  const extraEl = el("div", "cam-extra", "");
  const errEl = el("div", "cam-err", "");
  errEl.style.display = "none";
  body.appendChild(extraEl); body.appendChild(errEl);

  const acts = el("div", "cam-actions");
  const start = el("button", "btn primary sm", "▶ Запустить");
  const stop = el("button", "btn danger sm", "■ Остановить");
  const shot = el("button", "btn sm icon", "📸");
  shot.title = "Снимок кадра";
  const open = el("button", "btn sm icon", "⛶");
  open.title = "Открыть камеру";
  start.addEventListener("click", () => camAction("start", c.index, start));
  stop.addEventListener("click", () => camAction("stop", c.index, stop));
  shot.addEventListener("click", () => openSnapshot(currentCam(c.index)));
  open.addEventListener("click", () => openCameraModal(currentCam(c.index)));
  acts.appendChild(start); acts.appendChild(stop); acts.appendChild(shot); acts.appendChild(open);
  body.appendChild(acts);

  card.appendChild(liveWrap); card.appendChild(body);
  card._r = {stateEl, liveBadge, img, ph, pt, ps, resEl, fpsEl, srcEl,
             framesEl, procEl, uptimeEl, eventsEl, extraEl, errEl,
             start, stop, shot, nameEl};
  img.addEventListener("error", () => {
    img.classList.remove("on");
    ph.style.display = "";
    pt.textContent = "поток недоступен";
    ps.textContent = "проверь камеру и статус";
  });
  img.addEventListener("load", () => { img.classList.add("on"); ph.style.display = "none"; });
  return card;
}

function currentCam(index){
  return cameras.find(c => Number(c.index) === Number(index)) || {index};
}

function updateCamCard(card, c, canControl){
  const r = card._r;
  if (!r) return;
  const st = CAM_STATE[c.status] || [c.status || "—", "st-stopped"];
  r.stateEl.textContent = st[0];
  r.stateEl.className = "cam-state " + st[1];
  const running = c.status === "running" || c.status === "starting";
  card.classList.toggle("is-live", running);

  r.nameEl.textContent = c.name || ("Камера " + (c.index + 1));
  r.resEl.textContent = (c.width && c.height) ? c.width + "×" + c.height : "—";
  r.fpsEl.textContent = (c.proc_fps ? Number(c.proc_fps).toFixed(1) : "0") + " fps";
  r.srcEl.textContent = SRC_KIND[c.source_kind] || c.source_kind || "—";
  r.framesEl.textContent = fmtNum(c.frames || 0);
  r.procEl.textContent = c.proc_fps ? Number(c.proc_fps).toFixed(1) + " к/с" : "—";
  r.uptimeEl.textContent = c.uptime ? fmtDuration(c.uptime) : "—";
  r.eventsEl.textContent = String(c.events_24h || 0);

  const bits = [];
  if (c.fps) bits.push(Number(c.fps).toFixed(0) + " fps потока");
  if (c.recording) bits.push("запись" + (c.recording_mode ? ": " + c.recording_mode : ""));
  if (!c.enabled) bits.push("выключена в конфиге");
  if (c.last_event_ts) bits.push("событие " + ago(c.last_event_ts));
  if (c.last_frame_ts) bits.push("кадр " + ago(c.last_frame_ts));
  r.extraEl.textContent = bits.join(" · ");
  r.errEl.textContent = c.error ? "⚠ " + c.error : "";
  r.errEl.style.display = c.error ? "" : "none";

  r.start.disabled = !canControl || running || c.status === "stopping";
  r.stop.disabled = !canControl || !running;
  r.shot.disabled = !canControl || !c.has_image;

  const wantLive = canControl && running && !!c.has_image &&
                   activeTab === "cameras" && !document.hidden;
  if (wantLive){
    if (r.img.dataset.idx !== String(c.index) || !r.img.getAttribute("src")){
      r.img.dataset.idx = String(c.index);
      r.img.src = liveUrl(c.index);
    }
    r.ph.style.display = "none";
  } else {
    if (r.img.getAttribute("src")){
      r.img.removeAttribute("src");
      r.img.classList.remove("on");
    }
    r.img.dataset.idx = "";
    r.ph.style.display = "";
    r.pt.textContent = canControl ? (running ? "подключаемся…" : (c.status === "error" ? "нет сигнала" : "камера не работает"))
                                  : "только просмотр";
    r.ps.textContent = canControl ? (running ? "первый кадр скоро появится" : (c.status === "error" ? "смотри текст ошибки" : "нажми «Запустить»"))
                                  : "панель запущена отдельно от камеры";
  }
}

function applyCameras(items, canControl, stale){
  cameras = items || [];
  controllable = !!canControl;
  staleStatus = !!stale;
  const box = $("#camList");
  const sig = cameras.map(c => c.index).join(",");
  if (box.dataset.sig !== sig){
    box.dataset.sig = sig;
    box.innerHTML = "";
    camCards.clear();
    cameras.forEach(c => {
      const card = makeCamCard(c);
      camCards.set(String(c.index), card);
      box.appendChild(card);
    });
  }
  cameras.forEach(c => updateCamCard(camCards.get(String(c.index)), c, controllable));
  if (!cameras.length && !box.firstChild){
    const e = el("div", "empty");
    e.appendChild(el("div", "big", "🎥"));
    e.appendChild(el("div", "", "В конфиге нет камер"));
    e.appendChild(el("div", "hint", "Добавь cameras: в config/config.yaml"));
    box.appendChild(e);
  }

  const note = $("#camNote");
  note.style.display = controllable ? "none" : "block";
  note.textContent = stale
    ? "Нет свежих данных от процесса камеры: статусы могли устареть. Запусти python main.py — панель обновится сама."
    : "Панель запущена отдельным процессом: живые превью и кнопки доступны только вместе с камерой — python main.py --web.";

  updateHero(cameras);
  $("#camBadge").textContent = cameras.length ? runningCount() + "/" + cameras.length : "";

  const anyRunning = cameras.some(c => c.status === "running" || c.status === "starting");
  const anyStopped = cameras.some(c => c.status !== "running" && c.status !== "starting" && c.status !== "stopping");
  $("#startAll").disabled = !controllable || !anyStopped;
  $("#stopAll").disabled = !controllable || !anyRunning;
  renderSysCameras(cameras);
  updateOpenCameraModal();
}

function runningCount(){
  return cameras.filter(c => c.status === "running" || c.status === "starting").length;
}

function updateHero(list){
  const total = list.length;
  const running = runningCount();
  const events = list.reduce((a,c) => a + (c.events_24h || 0), 0);
  const fpsVals = list.filter(c => Number(c.proc_fps) > 0).map(c => Number(c.proc_fps));
  const avg = fpsVals.length ? fpsVals.reduce((a,b) => a+b, 0) / fpsVals.length : 0;
  const frames = list.reduce((a,c) => a + (Number(c.frames) || 0), 0);
  $("#heroRun").textContent = running;
  $("#heroTotal").textContent = total;
  $("#heroEvents").textContent = fmtNum(events);
  $("#heroFps").innerHTML = "";
  $("#heroFps").appendChild(document.createTextNode(avg ? avg.toFixed(1) : "0"));
  $("#heroFps").appendChild(el("small", "", "fps"));
  $("#heroFrames").textContent = fmtNum(frames);
}

async function loadCameras(){
  try{
    const d = await api("/api/cameras");
    if (d.stream){
      const changed = Number(d.stream.fps) !== Number(streamCfg.fps) ||
                      Number(d.stream.quality) !== Number(streamCfg.quality);
      streamCfg = {fps: Number(d.stream.fps) || 10, quality: Number(d.stream.quality) || 70};
      if (changed) refreshLiveStreams();
    }
    applyCameras(d.cameras, d.controllable, d.stale);
    setOnline(true);
  }catch(err){
    setOnline(false);
    $("#camBadge").textContent = "";
  }
}

function refreshLiveStreams(){
  cameras.forEach(c => {
    const card = camCards.get(String(c.index));
    const img = card && card._r && card._r.img;
    if (img && img.getAttribute("src")) img.src = liveUrl(c.index);
  });
}

function syncLiveStreams(){
  cameras.forEach(c => updateCamCard(camCards.get(String(c.index)), c, controllable));
}

/* ---------- снимок / модалка камеры ---------- */
function openSnapshot(c){
  if (!c || c.index === undefined) return;
  modalKind = "event";
  $("#mBadge").className = "badge car";
  $("#mBadge").textContent = "СНИМОК";
  $("#mTitle").textContent = c.name || "Камера";
  const body = $("#mBody"); body.innerHTML = "";
  const view = el("div", "cam-view");
  const img = el("img");
  img.alt = "Снимок с камеры";
  img.src = "/api/cameras/snapshot?index=" + encodeURIComponent(c.index) + "&quality=90&t=" + Date.now();
  view.appendChild(img);
  body.appendChild(view);
  const meta = el("div", "meta");
  const add = (k,v) => {
    if (v === undefined || v === null || v === "") return;
    const s = el("span"); s.appendChild(el("b","",k+": ")); s.appendChild(document.createTextNode(v));
    meta.appendChild(s);
  };
  add("Камера", c.name);
  add("Разрешение", (c.width && c.height) ? c.width + "×" + c.height : "");
  add("Кадров", String(c.frames || 0));
  add("Последний кадр", ago(c.last_frame_ts));
  add("Качество", "JPEG 90");
  body.appendChild(meta);
  const acts = el("div", "cam-actions");
  const dl = el("a", "btn", "⬇ Скачать JPEG");
  dl.href = "/api/cameras/snapshot?index=" + encodeURIComponent(c.index) + "&quality=95&download=1";
  dl.setAttribute("download", "");
  acts.appendChild(dl);
  body.appendChild(acts);
  $("#modal").classList.add("open");
}

function openCameraModal(c){
  if (!c || c.index === undefined) return;
  modalKind = "camera";
  modalCamIndex = Number(c.index);
  $("#mBadge").className = "badge car";
  $("#mBadge").textContent = "LIVE";
  $("#mTitle").textContent = c.name || ("Камера " + (c.index + 1));
  const body = $("#mBody"); body.innerHTML = "";

  const grid = el("div", "cam-modal");
  const view = el("div", "cam-view");
  const img = el("img"); img.alt = "Живой поток";
  const ph = el("div", "cam-ph");
  ph.appendChild(el("div", "pi", "📷"));
  const pt = el("div", "pt", "нет сигнала");
  const ps = el("div", "ps", "камера остановлена или ещё не дала кадров");
  ph.appendChild(pt); ph.appendChild(ps);
  view.appendChild(img); view.appendChild(ph);
  img.addEventListener("load", () => { img.classList.add("on"); ph.style.display = "none"; });
  img.addEventListener("error", () => { img.classList.remove("on"); ph.style.display = ""; });
  const running = c.status === "running" || c.status === "starting";
  if (controllable && running && c.has_image){
    img.src = liveUrl(c.index);
    img.classList.add("on");
    ph.style.display = "none";
  }
  grid.appendChild(view);

  const side = el("div", "cam-body");
  side.style.padding = "0";
  const actions = el("div", "cam-actions");
  const start = el("button", "btn primary sm", "▶ Запустить");
  const stop = el("button", "btn danger sm", "■ Остановить");
  const shot = el("button", "btn sm", "📸 Снимок");
  const dl = el("a", "btn sm", "⬇ JPEG");
  dl.href = "/api/cameras/snapshot?index=" + encodeURIComponent(c.index) + "&quality=95&download=1";
  dl.setAttribute("download", "");
  start.addEventListener("click", () => camAction("start", c.index, start));
  stop.addEventListener("click", () => camAction("stop", c.index, stop));
  shot.addEventListener("click", () => openSnapshot(currentCam(c.index)));
  actions.appendChild(start); actions.appendChild(stop); actions.appendChild(shot); actions.appendChild(dl);
  side.appendChild(actions);

  const kv = el("dl", "kv");
  const addKV = (k) => {
    kv.appendChild(el("dt", "", k));
    const dd = el("dd", "", "—");
    dd.dataset.mk = k;
    kv.appendChild(dd);
    return dd;
  };
  addKV("Статус"); addKV("Разрешение"); addKV("fps потока"); addKV("fps обработки");
  addKV("Кадров"); addKV("Аптайм"); addKV("Источник"); addKV("События 24 ч");
  side.appendChild(kv);

  const mhead = el("div", "mhead");
  mhead.appendChild(el("h4", "", "Последние события"));
  const allBtn = el("button", "btn sm ghost", "Все события →");
  allBtn.addEventListener("click", () => { closeModal(); switchTab("events"); setCameraFilter(c.name); });
  mhead.appendChild(allBtn);
  side.appendChild(mhead);
  const eventsBox = el("div", "mini-events");
  eventsBox.appendChild(el("div", "hint", "Загрузка…"));
  side.appendChild(eventsBox);

  grid.appendChild(side);
  body.appendChild(grid);
  fillCamModal(c);
  loadCamModalEvents(c, eventsBox);
  $("#modal").classList.add("open");
}

function fillCamModal(c){
  if (!c) return;
  const kv = $("#mBody").querySelector(".kv");
  if (!kv) return;
  const set = (key, val) => {
    const dd = kv.querySelector('[data-mk="' + key + '"]');
    if (dd) dd.textContent = val;
  };
  const st = CAM_STATE[c.status] || [c.status || "—"];
  set("Статус", st[0]);
  set("Разрешение", (c.width && c.height) ? c.width + "×" + c.height : "—");
  set("fps потока", c.fps ? Number(c.fps).toFixed(1) : "—");
  set("fps обработки", c.proc_fps ? Number(c.proc_fps).toFixed(1) : "—");
  set("Кадров", (c.frames || 0).toLocaleString("ru-RU"));
  set("Аптайм", c.uptime ? fmtDuration(c.uptime) : "—");
  set("Источник", SRC_KIND[c.source_kind] || c.source_kind || "—");
  set("События 24 ч", String(c.events_24h || 0));

  const body = $("#mBody");
  const btns = body.querySelectorAll(".cam-actions .btn");
  if (btns.length >= 2){
    const running = c.status === "running" || c.status === "starting";
    btns[0].disabled = !controllable || running;
    btns[1].disabled = !controllable || !running;
    btns[2].disabled = !controllable || !c.has_image;
  }
  const img = body.querySelector(".cam-view img");
  const ph = body.querySelector(".cam-view .cam-ph");
  const pt = ph ? ph.querySelector(".pt") : null;
  const ps = ph ? ph.querySelector(".ps") : null;
  if (img && pt && ps){
    const live = controllable && (c.status === "running" || c.status === "starting") && c.has_image;
    if (live){
      if (!img.getAttribute("src")){
        img.src = liveUrl(c.index);
        img.classList.add("on");
      }
      if (ph) ph.style.display = "none";
    } else {
      if (img.getAttribute("src")){
        img.removeAttribute("src");
        img.classList.remove("on");
      }
      if (ph) ph.style.display = "";
      pt.textContent = controllable ? (c.status === "error" ? "нет сигнала" : "камера не работает")
                                    : "только просмотр";
      ps.textContent = controllable ? (c.status === "error" ? (c.error || "смотри лог") : "нажми «Запустить»")
                                    : "панель запущена отдельно от камеры";
    }
  }
}

function updateOpenCameraModal(){
  if (modalKind !== "camera" || modalCamIndex < 0) return;
  const c = cameras.find(x => Number(x.index) === modalCamIndex);
  if (!c) return;
  fillCamModal(c);
}

async function loadCamModalEvents(c, box){
  if (!box) return;
  try{
    const d = await api("/api/events", {limit:6, hours:168, camera:c.name});
    box.innerHTML = "";
    const events = d.events || [];
    if (!events.length){
      box.appendChild(el("div", "hint", "За последнюю неделю событий не было."));
      return;
    }
    events.forEach(e => box.appendChild(makeMiniEvent(e)));
  }catch(err){
    box.innerHTML = "";
    box.appendChild(el("div", "hint", "Не удалось загрузить события: " + err.message));
  }
}

function makeMiniEvent(e){
  const b = el("button", "mev");
  if (e.has_image){
    const img = el("img"); img.loading = "lazy"; img.alt = ""; img.src = mediaUrl(e, false);
    b.appendChild(img);
  } else {
    const ph = el("div", "ph"); ph.textContent = TYPE_ICON[e.type] || "📷";
    b.appendChild(ph);
  }
  const t = el("div", "mt");
  t.appendChild(el("b", "", e.title || TYPE_LABEL[e.type] || "Событие"));
  t.appendChild(el("span", "", (TYPE_LABEL[e.type] || e.type) + (e.subtitle ? " · " + e.subtitle : "")));
  b.appendChild(t);
  b.appendChild(el("span", "ma", fmtTime(e.ts)));
  b.addEventListener("click", () => openModal(e));
  return b;
}

function closeModal(){
  const img = $("#mBody").querySelector(".cam-view img");
  if (img){ img.removeAttribute("src"); }
  $("#modal").classList.remove("open");
  modalKind = null;
  modalCamIndex = -1;
  if (modalTimer){ clearInterval(modalTimer); modalTimer = null; }
}

/* ---------- история ---------- */
function makeCard(e){
  const card = el("article", "card");
  card.title = "Открыть событие";
  const thumb = el("div", "thumb");
  if (e.has_image){
    const img = el("img");
    img.loading = "lazy"; img.alt = e.title; img.src = mediaUrl(e, false);
    img.onerror = () => { img.remove(); thumb.appendChild(placeholder(e.type)); };
    thumb.appendChild(img);
  } else {
    thumb.appendChild(placeholder(e.type));
  }
  thumb.appendChild(el("span", "badge " + e.type, TYPE_LABEL[e.type] || e.type));
  const body = el("div", "cbody");
  body.appendChild(el("div", "ctitle", e.title || TYPE_LABEL[e.type] || "Событие"));
  if (e.subtitle) body.appendChild(el("div", "csub", e.subtitle));
  const foot = el("div", "cfoot");
  foot.appendChild(el("span", "cam", e.camera || "камера"));
  foot.appendChild(el("span", "", fmtTime(e.ts)));
  body.appendChild(foot);
  card.appendChild(thumb); card.appendChild(body);
  card.addEventListener("click", () => openModal(e));
  return card;
}
function placeholder(type){
  return el("div", "ph", TYPE_ICON[type] || "📷");
}
function render(){
  const list = $("#list");
  list.innerHTML = "";
  lastDate = null;
  if (!state.events.length){
    const box = el("div", "empty");
    box.appendChild(el("div", "big", "🗂"));
    box.appendChild(el("div", "", state.type === "all" ? "Событий за этот период пока нет" : "По этому фильтру ничего не найдено"));
    box.appendChild(el("div", "hint", "Измени период или запусти камеру — история появится здесь."));
    list.appendChild(box);
    $("#moreWrap").style.display = "none";
    return;
  }
  state.events.forEach(e => {
    const d = fmtDate(e.ts);
    if (d !== lastDate){
      lastDate = d;
      list.appendChild(el("div", "divider", d));
    }
    list.appendChild(makeCard(e));
  });
  $("#moreWrap").style.display = state.hasMore ? "flex" : "none";
}

function openModal(e){
  modalKind = "event";
  $("#mBadge").className = "badge " + e.type;
  $("#mBadge").textContent = TYPE_LABEL[e.type] || e.type;
  $("#mTitle").textContent = e.title || "Событие";
  const body = $("#mBody"); body.innerHTML = "";
  const meta = el("div", "meta");
  const add = (k,v) => {
    if (v === undefined || v === null || v === "") return;
    const s = el("span"); s.appendChild(el("b","",k+": ")); s.appendChild(document.createTextNode(v));
    meta.appendChild(s);
  };
  add("Время", fmtDateTime(e.ts));
  add("Камера", e.camera);
  if (e.score !== null && e.score !== undefined) add("Уверенность", Number(e.score).toFixed(2));
  if (e.subtitle) add("Детали", e.subtitle);
  body.appendChild(meta);
  if (e.has_image || e.has_crop){
    const duo = el("div", e.has_crop && e.has_image ? "duo" : "");
    if (e.has_image){
      const img = el("img"); img.src = mediaUrl(e, false); img.alt = e.title; duo.appendChild(img);
    }
    if (e.has_crop){
      const img = el("img"); img.src = mediaUrl(e, true); img.alt = "Вырезка"; duo.appendChild(img);
    }
    body.appendChild(duo);
  }
  $("#modal").classList.add("open");
}

async function loadStats(){
  try{
    const s = await api("/api/stats", {hours: state.hours});
    $("#sTotal").textContent = fmtNum(s.total);
    $("#sFaces").textContent = fmtNum(s.faces);
    $("#sCars").textContent = fmtNum(s.cars);
    $("#sAnimals").textContent = fmtNum(s.animals);
    $("#sPackages").textContent = fmtNum(s.packages);
    $("#sUnknown").textContent = "незнакомцев: " + fmtNum(s.unknown);
    const sel = $("#camera");
    const current = state.camera;
    sel.innerHTML = "";
    sel.appendChild(new Option("Все камеры", ""));
    (s.by_camera || []).forEach(c => {
      if (!c.camera) return;
      sel.appendChild(new Option(c.camera + " (" + c.cnt + ")", c.camera));
    });
    sel.value = current;
    setOnline(true);
  }catch(err){ setOnline(false); }
}

async function loadTimeline(){
  const hours = state.hours > 0 ? state.hours : 168;
  const buckets = hours <= 24 ? 24 : (hours <= 72 ? 36 : 48);
  try{
    const d = await api("/api/timeline", {hours, buckets});
    renderChart(d);
    $("#chartTitle").textContent = state.hours > 0
      ? "за последние " + (hours === 24 ? "24 часа" : (hours === 72 ? "3 дня" : "7 дней"))
      : "за последние 7 дней";
  }catch(err){
    $("#chart").innerHTML = "";
    $("#chartFoot").textContent = "график недоступен";
  }
}

function renderChart(d){
  const box = $("#chart");
  box.innerHTML = "";
  const items = d.items || [];
  const max = Math.max(1, d.max || 1);
  const width = items.length ? Math.max(1, (d.hours * 3600) / items.length) : 3600;
  const step = Math.max(1, Math.ceil(items.length / 8));
  items.forEach((it, i) => {
    const col = el("div", "bar-col");
    const stack = el("div", "bar-stack");
    const pct = Math.max(2, Math.round((it.total / max) * 100));
    stack.style.height = pct + "%";
    [["face", it.face], ["car", it.car], ["animal", it.animal], ["package", it.package]].forEach(([k, v]) => {
      if (!v) return;
      const seg = el("div", "seg " + k);
      seg.style.flex = v + " 1 0";
      stack.appendChild(seg);
    });
    const from = new Date(it.ts*1000);
    const to = new Date((it.ts + width)*1000);
    const fmt = (dt) => dt.toLocaleTimeString("ru-RU", {hour:"2-digit", minute:"2-digit"});
    stack.title = fmt(from) + "–" + fmt(to) + " · всего " + it.total +
      "\n👤 люди: " + it.face + "   🚗 машины: " + it.car +
      "\n🐾 животные: " + it.animal + "   📦 вещи: " + it.package;
    col.appendChild(stack);
    if (i % step === 0 || i === items.length - 1){
      col.appendChild(el("span", "bar-lbl", fmt(from)));
    } else {
      col.appendChild(el("span", "bar-lbl", ""));
    }
    box.appendChild(col);
  });
  const total = items.reduce((a,x) => a + (x.total || 0), 0);
  $("#chartFoot").textContent = "всего за период: " + total + " · пик: " + (d.max || 0) + " за интервал";
}

async function loadEvents(reset){
  if (reset){ state.offset = 0; state.events = []; }
  try{
    const data = await api("/api/events", {
      limit: state.limit, offset: state.offset, kind: state.type,
      camera: state.camera, q: state.q, hours: state.hours
    });
    state.events = state.events.concat(data.events || []);
    state.hasMore = !!data.has_more;
    state.offset = state.events.length;
    render();
    setOnline(true);
  }catch(err){
    toast("Не удалось загрузить события: " + err.message, "err");
    setOnline(false);
  }
}
async function refreshAll(){
  await Promise.all([loadStats(), loadEvents(true), loadTimeline()]);
}

/* ---------- система ---------- */
async function loadSystem(){
  try{
    const s = await api("/api/system");
    $("#sysVersion").textContent = "v" + s.version;
    $("#sysVersionBig").textContent = "v" + s.version;
    $("#sysUptime").textContent = fmtDuration(s.uptime);
    $("#sysDb").textContent = fmtBytes(s.db_bytes);
    const free = s.disk_free || 0, total = s.disk_total || 0;
    const usedPct = total ? Math.max(0, Math.min(100, (1 - free/total) * 100)) : 0;
    $("#sysDisk").textContent = fmtBytes(free);
    $("#sysDiskBar").style.width = usedPct.toFixed(0) + "%";
    $("#sysDiskText").textContent = total
      ? "занято " + usedPct.toFixed(0) + "% из " + fmtBytes(total)
      : "нет данных";
    $("#sysControllable").textContent = s.controllable
      ? "управление доступно: панель работает вместе с камерой"
      : "только просмотр: панель запущена отдельно";

    const f = s.features || {};
    const chips = [
      ["👤 Лица", !!f.face],
      ["🚗 Машины", !!f.car],
      ["🐾 Животные и вещи", !!f.objects],
      ["🎬 Запись", !!f.recording],
      ["🖥 Превью", !!f.preview],
      ["🔢 Номера: " + (f.plate_engine || "—"), !!f.plate_engine]
    ];
    const fbox = $("#sysFeatures");
    fbox.innerHTML = "";
    chips.forEach(([label, on]) => {
      const c = el("span", "chip " + (on ? "on" : "off"), label);
      fbox.appendChild(c);
    });
    setOnline(true);
  }catch(err){ setOnline(false); }
}

function renderSysCameras(list){
  const body = $("#sysCamTable");
  if (!body) return;
  if (!list || !list.length){
    body.innerHTML = "";
    const tr = el("tr");
    const td = el("td", "hint", "В конфиге нет камер");
    td.colSpan = 8;
    tr.appendChild(td); body.appendChild(tr);
    return;
  }
  body.innerHTML = "";
  list.forEach(c => {
    const tr = el("tr");
    tr.appendChild(el("td", "", "#" + (Number(c.index) + 1)));
    tr.appendChild(el("td", "", c.name || "—"));
    const st = CAM_STATE[c.status] || [c.status || "—", "st-stopped"];
    const td = el("td");
    td.appendChild(el("span", "cam-state " + st[1], st[0]));
    tr.appendChild(td);
    tr.appendChild(el("td", "", SRC_KIND[c.source_kind] || c.source_kind || "—"));
    tr.appendChild(el("td", "", (c.width && c.height) ? c.width + "×" + c.height : "—"));
    tr.appendChild(el("td", "", c.proc_fps ? Number(c.proc_fps).toFixed(1) + " fps" : "—"));
    tr.appendChild(el("td", "", c.uptime ? fmtDuration(c.uptime) : "—"));
    tr.appendChild(el("td", "", String(c.events_24h || 0)));
    body.appendChild(tr);
  });
}

/* ---------- настройки ---------- */
let settingsData = null;

function fieldControl(f, value, idx){
  const id = "setf_" + idx;
  const isBool = f.type === "bool";
  const wrap = el("div", "set-field" + (isBool ? " bool" : ""));
  const lab = el("label", null, f.label);
  lab.htmlFor = id;
  if (f.unit) lab.appendChild(el("span", "unit", f.unit));
  if (isBool){
    wrap.appendChild(lab);
    const cb = el("input");
    cb.type = "checkbox"; cb.id = id; cb.checked = !!value;
    cb.dataset.path = f.path; cb.dataset.kind = "bool";
    wrap.appendChild(cb);
  } else {
    wrap.appendChild(lab);
    let input;
    if (f.type === "select"){
      input = el("select");
      (f.options || []).forEach(opt => {
        const o = el("option", null, opt); o.value = opt;
        if (String(opt) === String(value)) o.selected = true;
        input.appendChild(o);
      });
    } else {
      input = el("input");
      if (f.type === "int" || f.type === "float"){
        input.type = "number";
        if (f.min !== undefined) input.min = f.min;
        if (f.max !== undefined) input.max = f.max;
        input.step = f.step !== undefined ? f.step : (f.type === "int" ? 1 : 0.01);
        input.value = value === undefined || value === null ? "" : value;
        input.inputMode = "decimal";
      } else if (f.type === "password"){
        input.type = "password"; input.value = value || "";
        input.autocomplete = "new-password";
      } else if (f.type === "time"){
        input.type = "time"; input.value = value || "";
      } else {
        input.type = "text";
        input.value = Array.isArray(value) ? value.join(", ") : (value ?? "");
      }
    }
    input.id = id;
    input.dataset.path = f.path;
    input.dataset.kind = f.type;
    wrap.appendChild(input);
  }
  if (f.help) wrap.appendChild(el("div", "fh", f.help));
  return wrap;
}

function camEditRow(c){
  const row = el("div", "cam-row");
  const name = el("input");
  name.type = "text"; name.placeholder = "Имя камеры"; name.value = c.name || "";
  name.dataset.cam = "name";
  const src = el("input");
  src.type = "text"; src.placeholder = "0 или rtsp://…";
  src.value = String(c.source ?? 0);
  src.dataset.cam = "source";
  const en = el("label", "cam-enabled");
  const cb = el("input");
  cb.type = "checkbox"; cb.checked = c.enabled !== false; cb.dataset.cam = "enabled";
  en.appendChild(cb);
  en.appendChild(document.createTextNode(" включена"));
  const rm = el("button", "rm", "×");
  rm.type = "button"; rm.title = "Удалить камеру";
  rm.addEventListener("click", () => row.remove());
  row.appendChild(name); row.appendChild(src);
  row.appendChild(en); row.appendChild(rm);
  return row;
}

function renderCamEditor(cams){
  const card = el("div", "set-card");
  card.appendChild(el("h3", null, "🎥 Камеры"));
  card.appendChild(el("p", null,
    "0 — веб-камера ноутбука. Для IP-камеры укажи rtsp://логин:пароль@IP:554/stream1. " +
    "Первый запуск с новой камерой может занять 10–30 секунд."));
  const list = el("div", "cam-edit-list"); list.id = "setCams";
  (cams || []).forEach(c => list.appendChild(camEditRow(c)));
  card.appendChild(list);
  const add = el("button", "btn ghost cam-add", "+ Добавить камеру");
  add.type = "button";
  add.addEventListener("click", () => {
    list.appendChild(camEditRow({name: "", source: 0, enabled: true}));
    const inputs = list.querySelectorAll(".cam-row:last-child input");
    if (inputs.length) inputs[0].focus();
  });
  card.appendChild(add);
  return card;
}

function renderSettings(d){
  const form = $("#setForm");
  form.innerHTML = "";
  $("#setPath").textContent = d.config_path || "config/config.yaml";
  form.appendChild(renderCamEditor(d.cameras || []));
  let idx = 0;
  (d.sections || []).forEach(section => {
    const fields = (d.fields || []).filter(f => f.section === section.id);
    if (!fields.length) return;
    const card = el("div", "set-card");
    card.appendChild(el("h3", null, (section.icon || "•") + " " + section.title));
    if (section.hint) card.appendChild(el("p", null, section.hint));
    const grid = el("div", "set-grid");
    fields.forEach(f => {
      grid.appendChild(fieldControl(f, d.values ? d.values[f.path] : f.default, idx++));
    });
    card.appendChild(grid);
    form.appendChild(card);
  });
  const foot = el("div", "set-foot");
  foot.appendChild(el("span", "hint",
    "Файл: " + (d.config_path || "config/config.yaml") + " · резервная копия: .bak"));
  form.appendChild(foot);
}

function collectSettings(){
  const values = {};
  $$("#setForm [data-path]").forEach(input => {
    const kind = input.dataset.kind;
    let value;
    if (kind === "bool") value = input.checked;
    else if (kind === "int" || kind === "float")
      value = input.value === "" ? 0 : Number(input.value);
    else value = input.value;
    values[input.dataset.path] = value;
  });
  const cameras = [];
  $$("#setCams .cam-row").forEach(row => {
    const name = row.querySelector('[data-cam="name"]');
    const src = row.querySelector('[data-cam="source"]');
    const en = row.querySelector('[data-cam="enabled"]');
    if (!name && !src) return;
    cameras.push({
      name: name ? name.value.trim() : "",
      source: src ? src.value.trim() : "0",
      enabled: en ? en.checked : true
    });
  });
  return {values, cameras};
}

async function loadSettings(force){
  if (settingsData && !force) return;
  try{
    const d = await api("/api/settings");
    settingsData = d;
    renderSettings(d);
  }catch(err){
    $("#setForm").innerHTML = "";
    $("#setForm").appendChild(el("div", "note",
      "Не удалось загрузить настройки: " + err.message));
  }
}

async function saveSettings(btn){
  const {values, cameras} = collectSettings();
  const restart = $("#setRestart").checked;
  const hint = $("#setHint");
  hint.style.display = "none";
  btn.disabled = true;
  btn.textContent = "Сохраняю…";
  try{
    const d = await apiPost("/api/settings", {values, cameras, restart});
    toast(d.message || "Настройки сохранены", "ok");
    settingsData = d.settings || settingsData;
    if (settingsData){
      if (settingsData.stream) streamCfg = settingsData.stream;
      renderSettings(settingsData);
    }
    if (d.restarted) setTimeout(loadCameras, 1400);
    else refreshLiveStreams();
  }catch(err){
    const msg = (err.data && err.data.errors)
      ? err.data.errors.join("; ") : err.message;
    hint.textContent = "Не сохранено: " + msg;
    hint.style.display = "";
    toast("Проверь поля: " + msg, "err");
  }finally{
    btn.disabled = false;
    btn.textContent = "💾 Сохранить и применить";
  }
}

$("#setSave").addEventListener("click", e => saveSettings(e.currentTarget));
$("#setReload").addEventListener("click", () => {
  settingsData = null;
  loadSettings(true);
  toast("Значения перечитаны из конфига");
});

/* ---------- вкладки / фильтры ---------- */
function switchTab(name){
  activeTab = name;
  $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === name));
  $("#camerasView").style.display = name === "cameras" ? "" : "none";
  $("#eventsView").style.display = name === "events" ? "" : "none";
  $("#settingsView").style.display = name === "settings" ? "" : "none";
  $("#systemView").style.display = name === "system" ? "" : "none";
  $("#mainFoot").style.display = (name === "system" || name === "settings") ? "none" : "";
  if (name === "cameras"){
    loadCameras();
    syncLiveStreams();
  } else {
    syncLiveStreams();
    if (name === "events") refreshAll();
    if (name === "settings") loadSettings(false);
    if (name === "system") loadSystem();
  }
}

function setCameraFilter(name){
  state.camera = name || "";
  const sel = $("#camera");
  if (sel) sel.value = state.camera;
  refreshAll();
}

$("#type").addEventListener("change", e => { state.type = e.target.value; refreshAll(); });
$("#camera").addEventListener("change", e => { state.camera = e.target.value; refreshAll(); });
$("#hours").addEventListener("change", e => { state.hours = Number(e.target.value); refreshAll(); });
let searchTimer = null;
$("#search").addEventListener("input", e => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.q = e.target.value.trim(); refreshAll(); }, 320);
});
$("#refresh").addEventListener("click", refreshAll);
$("#more").addEventListener("click", () => loadEvents(false));
$("#mClose").addEventListener("click", closeModal);
$("#modal").addEventListener("click", e => { if (e.target === $("#modal")) closeModal(); });
document.addEventListener("keydown", e => {
  if (e.key === "Escape") closeModal();
  if (e.target && /input|select|textarea/i.test(e.target.tagName)) return;
  const map = {"1":"cameras", "2":"events", "3":"settings", "4":"system"};
  if (map[e.key]) switchTab(map[e.key]);
});

$("#startAll").addEventListener("click", e => camAction("start_all", null, e.currentTarget));
$("#stopAll").addEventListener("click", e => camAction("stop_all", null, e.currentTarget));
$("#camRefresh").addEventListener("click", loadCameras);

$("#autoRefresh").addEventListener("change", e => {
  autoOn = e.target.checked;
  if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
  if (autoOn) autoTimer = setInterval(autoRefreshTick, 5000);
  toast(autoOn ? "Автообновление включено" : "Автообновление выключено");
});

$$(".tab").forEach(tab => tab.addEventListener("click", () => switchTab(tab.dataset.tab)));

document.addEventListener("visibilitychange", () => { syncLiveStreams(); });

function autoRefreshTick(){
  if (!autoOn) return;
  if (activeTab === "cameras") loadCameras();
  else if (activeTab === "events"){
    const box = $("#list");
    if (box) loadEvents(true);
    loadStats();
    loadTimeline();
  } else if (activeTab === "system") loadSystem();
}

function tick(){
  $("#clock").textContent = new Date().toLocaleTimeString("ru-RU");
  const heroFrames = document.querySelectorAll("[data-ago]");
  heroFrames.forEach(n => { n.textContent = ago(Number(n.dataset.ago)); });
}
setInterval(tick, 1000); tick();

switchTab("cameras");
setTimeout(loadSystem, 300);
autoTimer = setInterval(autoRefreshTick, 5000);
</script>
</body>
</html>
"""


def _one(params: dict, name: str, default: str = "") -> str:
    values = params.get(name)
    return str(values[0]) if values else default


def _int_param(params: dict, name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(_one(params, name, str(default)))))
    except (TypeError, ValueError):
        return default


def _float_param(params: dict, name: str, default: float) -> float:
    try:
        return float(_one(params, name, str(default)))
    except (TypeError, ValueError):
        return default


class _WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, *, cfg: Config, db: Database, token: str,
                 title: str, events_dir: Path, owns_db: bool,
                 manager=None, control: bool = True):
        super().__init__(address, handler)
        self.cfg = cfg
        self.db = db
        self.token = token
        self.title = title
        self.events_dir = events_dir
        self.owns_db = owns_db
        self.manager = manager
        self.control = bool(control)
        self.started_at = time.time()
        self._closing = threading.Event()

    def server_close(self) -> None:
        self._closing.set()
        super().server_close()
        if self.owns_db:
            try:
                self.db.close()
            except Exception:
                pass

    def handle_error(self, request, client_address) -> None:
        """Не засорять лог, когда браузер закрыл вкладку или оборвал keep-alive."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError,
                            BrokenPipeError, TimeoutError)):
            log.debug("Соединение с %s закрыто клиентом", client_address)
            return
        log.warning("Ошибка обработки запроса от %s: %s", client_address, exc)


class EventsHandler(BaseHTTPRequestHandler):
    server_version = "MLCameraWeb/1.0"
    protocol_version = "HTTP/1.1"

    # ---------- служебное ----------
    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    @property
    def _srv(self) -> _WebServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        self._handle(head=False)

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle(head=True)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        self._body = self._read_body()
        self._set_auth_cookie = False
        if not self._authorized(params):
            self._unauthorized(head=False)
            return
        if not self._mutation_allowed():
            self._send_json({"ok": False,
                             "error": "Запрос отклонён: нет заголовка X-MLCamera"},
                            head=False, status=403)
            return
        try:
            if path == "/api/settings":
                self._api_settings_post(params)
            elif path.startswith("/api/cameras/"):
                self._api_camera_action(path.rsplit("/", 1)[-1], params)
            else:
                self._send_bytes(404, "text/plain; charset=utf-8", b"Not found", False)
        except BrokenPipeError:
            pass
        except Exception:
            log.exception("Ошибка веб-панели (POST): %s", self.path)
            self._send_bytes(500, "text/plain; charset=utf-8", b"Internal error", False)

    def _read_body(self) -> bytes:
        """Прочитать тело запроса (JSON настроек) с ограничением размера."""
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if length <= 0:
            return b""
        if length > 256 * 1024:
            return b""
        return self.rfile.read(length)

    def _discard_body(self) -> None:
        """Прочитать тело запроса, чтобы не ломать keep-alive."""
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            length = 0
        if 0 < length <= 1024 * 1024:
            self.rfile.read(length)

    def _mutation_allowed(self) -> bool:
        """CSRF-защита: свой заголовок + совпадающий Origin (если он есть)."""
        if self.headers.get("X-MLCamera") != "1":
            return False
        origin = self.headers.get("Origin")
        if origin:
            try:
                if urlsplit(origin).netloc != (self.headers.get("Host") or ""):
                    return False
            except ValueError:
                return False
        return True

    def _handle(self, head: bool) -> None:
        parsed = urlsplit(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        self._set_auth_cookie = False
        if not self._authorized(params):
            self._unauthorized(head)
            return
        try:
            if path in ("/", "/index.html"):
                self._send_html(head)
            elif path == "/health":
                self._send_json({"ok": True}, head)
            elif path == "/api/events":
                self._api_events(params, head)
            elif path == "/api/stats":
                self._api_stats(params, head)
            elif path == "/api/timeline":
                self._api_timeline(params, head)
            elif path == "/api/system":
                self._api_system(head)
            elif path == "/api/settings":
                self._api_settings_get(head)
            elif path == "/api/cameras":
                self._api_cameras(head)
            elif path == "/api/cameras/snapshot":
                self._api_camera_snapshot(params, head)
            elif path == "/api/cameras/stream":
                self._api_camera_stream(params, head)
            elif path == "/media":
                self._media(params, head)
            elif path == "/favicon.ico":
                self._send_bytes(204, "image/x-icon", b"", head)
            else:
                self._send_bytes(404, "text/plain; charset=utf-8", b"Not found", head)
        except BrokenPipeError:
            pass
        except Exception:
            log.exception("Ошибка веб-панели: %s", self.path)
            self._send_bytes(500, "text/plain; charset=utf-8", b"Internal error", head)

    # ---------- авторизация ----------
    def _cookie_token(self) -> str:
        raw = self.headers.get("Cookie")
        if not raw:
            return ""
        try:
            jar = SimpleCookie()
            jar.load(raw)
            morsel = jar.get("mlc_token")
            return morsel.value if morsel else ""
        except Exception:
            return ""

    def _authorized(self, params: dict) -> bool:
        token = self._srv.token
        if not token:
            return True
        supplied = _one(params, "token") or self._cookie_token() or self.headers.get("X-Token", "")
        if supplied and secrets.compare_digest(str(supplied), token):
            self._set_auth_cookie = True
            return True
        return False

    def _unauthorized(self, head: bool) -> None:
        body = ("Доступ к веб-панели закрыт.\n"
                "Открой адрес с параметром ?token=ВАШ_ТОКЕН (см. web.token в конфиге).\n")
        self._send_bytes(401, "text/plain; charset=utf-8", body.encode("utf-8"), head)

    # ---------- ответы ----------
    def _common_headers(self, ctype: str, length: int, cache: str = "no-store",
                        csp: bool = False) -> None:
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if csp:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; "
                "form-action 'none'; frame-ancestors 'none'",
            )
        if self._set_auth_cookie:
            cookie = f"mlc_token={self._srv.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000"
            self.send_header("Set-Cookie", cookie)

    def _send_bytes(self, status: int, ctype: str, body: bytes, head: bool,
                    cache: str = "no-store", csp: bool = False,
                    extra: Optional[dict] = None) -> None:
        self.send_response(status)
        self._common_headers(ctype, len(body), cache, csp)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if not head and body:
            self.wfile.write(body)

    def _send_json(self, payload, head: bool, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", body, head)

    def _send_html(self, head: bool) -> None:
        title = html.escape(self._srv.title, quote=True)
        body = _HTML_PAGE.replace("__TITLE__", title).encode("utf-8")
        self._send_bytes(200, "text/html; charset=utf-8", body, head, csp=True)

    # ---------- API ----------
    def _api_events(self, params: dict, head: bool) -> None:
        limit = _int_param(params, "limit", 30, 1, 200)
        offset = _int_param(params, "offset", 0, 0, 1_000_000)
        kind = _one(params, "kind", "all") or "all"
        if kind not in ("all", "object", "face", "car", "animal", "package"):
            kind = "all"
        camera = _one(params, "camera") or None
        q = _one(params, "q") or None
        hours = _float_param(params, "hours", 24.0)
        events, has_more = self._srv.db.query_events(
            limit=limit, offset=offset, kind=kind, camera=camera, q=q, hours=hours)
        self._send_json({"events": events, "has_more": has_more,
                         "limit": limit, "offset": offset}, head)

    def _api_stats(self, params: dict, head: bool) -> None:
        hours = _float_param(params, "hours", 24.0)
        stats = self._srv.db.stats(hours=hours)
        self._send_json(stats, head)

    def _api_timeline(self, params: dict, head: bool) -> None:
        hours = _float_param(params, "hours", 24.0)
        buckets = _int_param(params, "buckets", 24, 4, 96)
        self._send_json(self._srv.db.timeline(hours=hours, buckets=buckets), head)

    def _api_system(self, head: bool) -> None:
        cfg = self._srv.cfg
        db_path = cfg.database.resolve(cfg.database.path)
        db_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(db_path) + suffix)
            try:
                db_bytes += path.stat().st_size
            except OSError:
                pass
        disk_free = disk_total = 0
        probe = self._srv.events_dir
        try:
            usage = shutil.disk_usage(probe if probe.exists() else probe.parent)
            disk_free, disk_total = int(usage.free), int(usage.total)
        except OSError:
            pass

        manager = self._srv.manager
        if manager is not None:
            cams = manager.list()
        else:
            cams = [c for c in (read_status(cfg).get("cameras") or [])
                    if isinstance(c, dict)]
        running = sum(1 for c in cams
                      if c.get("status") in ("running", "starting"))
        errors = sum(1 for c in cams if c.get("status") == "error")

        def section(name: str, *keys: str) -> dict:
            data = cfg.get(name, None)
            if not isinstance(data, dict):
                return {}
            return {key: data.get(key) for key in keys if key in data}

        car = cfg.get("car", None)
        self._send_json({
            "version": __version__,
            "uptime": round(max(0.0, time.time() - self._srv.started_at), 0),
            "db_bytes": db_bytes,
            "disk_free": disk_free,
            "disk_total": disk_total,
            "cameras": {"total": len(cams), "running": running, "errors": errors},
            "features": {
                "face": bool(section("face", "enabled").get("enabled", True)),
                "car": bool(section("car", "enabled").get("enabled", True)),
                "objects": bool(section("objects", "enabled").get("enabled", True)),
                "recording": bool(section("recording", "enabled").get("enabled", False)),
                "preview": bool(section("ui", "preview_enabled").get("preview_enabled", False)),
                "plate_engine": (car or {}).get("plate_engine", "") if isinstance(car, dict) else "",
            },
            "controllable": bool(self._srv.manager is not None and self._srv.control),
        }, head)

    # ---------- настройки ----------
    def _api_settings_get(self, head: bool) -> None:
        cfg = self._srv.cfg
        payload = settings_payload(cfg)
        payload["can_apply"] = bool(self._srv.manager is not None and self._srv.control)
        payload["version"] = __version__
        self._send_json(payload, head)

    def _api_settings_post(self, params: dict) -> None:
        cfg = self._srv.cfg
        try:
            body = json.loads((getattr(self, "_body", b"") or b"{}").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"ok": False, "error": "Некорректный JSON"},
                            head=False, status=400)
            return
        if not isinstance(body, dict):
            self._send_json({"ok": False, "error": "Ожидался объект JSON"},
                            head=False, status=400)
            return

        old_web = dict(cfg.get("web", {}) or {})
        values = body.get("values") or {}
        cameras = body.get("cameras", None)
        if not isinstance(values, dict):
            self._send_json({"ok": False, "error": "Поле values должно быть объектом"},
                            head=False, status=400)
            return
        if cameras is not None and not isinstance(cameras, list):
            self._send_json({"ok": False, "error": "Поле cameras должно быть списком"},
                            head=False, status=400)
            return

        try:
            save_settings(cfg, values, cameras)
        except SettingsError as exc:
            self._send_json({"ok": False, "errors": exc.errors,
                             "error": "; ".join(exc.errors)},
                            head=False, status=400)
            return
        except OSError as exc:
            log.exception("Не удалось сохранить настройки")
            self._send_json({"ok": False, "error": f"Не удалось записать конфиг: {exc}"},
                            head=False, status=500)
            return

        manager = self._srv.manager
        applied = apply_live(cfg, manager) if manager is not None else []
        restart = str(body.get("restart", "1")).lower() not in ("0", "false", "no", "")
        restarted = False
        if manager is not None and self._srv.control and restart:
            reload_cameras_async(manager, start=True)
            restarted = True

        notes: list[str] = []
        if restarted:
            notes.append("камеры перезапускаются — новый FPS/разрешение применятся "
                         "через несколько секунд")
        elif manager is not None:
            notes.append("сохранено; FPS/разрешение применятся после перезапуска камеры")
        else:
            notes.append("сохранено; панель запущена отдельно — перезапусти камеру")
        if applied:
            notes.append("рамки и «чертёж лица» обновлены сразу")
        new_web = cfg.get("web", {}) or {}
        for key, label in (("host", "адрес"), ("port", "порт"), ("token", "пароль")):
            if str(old_web.get(key, "")) != str(new_web.get(key, "")):
                notes.append(f"{label} панели изменится после перезапуска main.py")

        self._send_json({
            "ok": True,
            "message": "; ".join(notes),
            "restarted": restarted,
            "config_path": str(cfg.config_path),
            "settings": settings_payload(cfg),
        }, head=False)

    # ---------- камеры ----------
    def _cameras_payload(self) -> dict:
        manager = self._srv.manager
        stats = self._srv.db.camera_stats(24)
        stream = {
            "fps": int(_web_value(self._srv.cfg, "stream_fps", 10)),
            "quality": int(_web_value(self._srv.cfg, "stream_quality", 70)),
        }
        if manager is not None:
            cameras = []
            for cam in manager.list():
                row = stats.get(str(cam.get("name", ""))) or {}
                cam["events_24h"] = int(row.get("events", 0))
                cam["last_event_ts"] = float(row.get("last_ts") or 0)
                cameras.append(cam)
            return {"controllable": bool(self._srv.control),
                    "stale": False,
                    "cameras": cameras,
                    "stream": stream,
                    "updated_at": time.time()}
        data = read_status(self._srv.cfg)
        cameras = []
        for cam in data.get("cameras") or []:
            if not isinstance(cam, dict):
                continue
            cam = dict(cam)
            # отдельный процесс не может сделать снимок/стрим
            cam["has_image"] = False
            row = stats.get(str(cam.get("name", ""))) or {}
            cam["events_24h"] = int(row.get("events", 0))
            cam["last_event_ts"] = float(row.get("last_ts") or 0)
            cameras.append(cam)
        updated_at = float(data.get("updated_at") or 0)
        # процесс камеры пишет статус раз в 2 с; давно нет записи — нет связи
        stale = bool(not updated_at or (time.time() - updated_at) > 15)
        return {"controllable": False, "stale": stale, "cameras": cameras,
                "stream": stream, "updated_at": updated_at}

    def _api_cameras(self, head: bool) -> None:
        self._send_json(self._cameras_payload(), head)

    def _api_camera_action(self, action: str, params: dict) -> None:
        manager = self._srv.manager
        if manager is None or not self._srv.control:
            self._send_json(
                {"ok": False,
                 "error": "Управление доступно, только когда панель запущена "
                          "вместе с камерой: python main.py --web"},
                head=False, status=409)
            return
        if action in ("start_all", "stop_all"):
            if action == "start_all":
                manager.start_all()
            else:
                manager.stop_all()
            self._send_json({"ok": True, "message": "Готово",
                             "cameras": manager.list()}, head=False)
            return
        if action not in ("start", "stop"):
            self._send_bytes(404, "text/plain; charset=utf-8", b"Not found", False)
            return
        index = _int_param(params, "index", -1, -1, 1000)
        if index < 0:
            self._send_json({"ok": False, "error": "Не указан index камеры"},
                            head=False, status=400)
            return
        ok, message = ((manager.start(index) if action == "start"
                        else manager.stop(index)))
        self._send_json({"ok": ok, "message": message,
                         "cameras": manager.list()},
                        head=False, status=200 if ok else 409)

    def _api_camera_snapshot(self, params: dict, head: bool) -> None:
        manager = self._srv.manager
        if manager is None or not self._srv.control:
            self._send_bytes(
                409, "text/plain; charset=utf-8",
                "Снимок доступен только вместе с камерой".encode("utf-8"), head)
            return
        index = _int_param(params, "index", -1, -1, 1000)
        quality = _int_param(params, "quality", 85, 30, 95)
        data = manager.snapshot(index, quality=quality) if index >= 0 else None
        if not data:
            self._send_bytes(404, "text/plain; charset=utf-8", b"No frame", head)
            return
        filename = f"mlcamera-{index}-{int(time.time())}.jpg"
        disposition = "attachment" if _one(params, "download") in ("1", "true", "yes") else "inline"
        self._send_bytes(
            200, "image/jpeg", data, head, cache="no-store",
            extra={"Content-Disposition": f'{disposition}; filename="{filename}"'})

    def _api_camera_stream(self, params: dict, head: bool) -> None:
        """MJPEG-поток: браузер показывает его как обычную картинку."""
        manager = self._srv.manager
        if manager is None or not self._srv.control:
            self._send_bytes(
                409, "text/plain; charset=utf-8",
                "Поток доступен только вместе с камерой".encode("utf-8"), head)
            return
        index = _int_param(params, "index", -1, -1, 1000)
        if index < 0:
            self._send_bytes(400, "text/plain; charset=utf-8", b"Bad index", head)
            return
        fps = _int_param(params, "fps",
                         int(_web_value(self._srv.cfg, "stream_fps", 10)), 1, 30)
        quality = _int_param(params, "quality",
                            int(_web_value(self._srv.cfg, "stream_quality", 70)), 30, 95)
        boundary = "mlcframe" + secrets.token_hex(6)
        self.send_response(200)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        if head:
            return

        part = ("--" + boundary + "\r\n"
                "Content-Type: image/jpeg\r\n").encode("ascii")
        delay = 1.0 / fps
        empty_since = time.time()
        try:
            while not self._srv._closing.is_set():
                cam = next((c for c in manager.list()
                            if int(c.get("index", -1)) == index), None)
                if cam is None or cam.get("status") not in ("running", "starting"):
                    break
                data = manager.snapshot(index, quality=quality)
                if not data:
                    if time.time() - empty_since > 10:
                        break
                    time.sleep(0.2)
                    continue
                empty_since = time.time()
                self.wfile.write(part)
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError,
                TimeoutError, OSError):
            # вкладку закрыли или поток остановили — это нормально
            pass

    # ---------- фото ----------
    def _media(self, params: dict, head: bool) -> None:
        kind = _one(params, "type")
        if kind not in _ALLOWED_TYPES:
            self._send_bytes(400, "text/plain; charset=utf-8", b"Bad type", head)
            return
        try:
            event_id = int(_one(params, "id", "0"))
        except ValueError:
            event_id = 0
        if event_id <= 0:
            self._send_bytes(400, "text/plain; charset=utf-8", b"Bad id", head)
            return
        crop = _one(params, "crop") in ("1", "true", "yes")
        raw = self._srv.db.get_media_path(kind, event_id, crop=crop)
        if not raw:
            self._send_bytes(404, "text/plain; charset=utf-8", b"No image", head)
            return
        path = Path(raw)
        if not path.is_absolute():
            path = self._srv.cfg.resolve(raw)
        try:
            path = path.resolve()
            path.relative_to(self._srv.events_dir)
        except (OSError, ValueError):
            log.warning("Попытка отдать файл вне data/events: %s", raw)
            self._send_bytes(403, "text/plain; charset=utf-8", b"Forbidden", head)
            return
        if not path.is_file():
            self._send_bytes(404, "text/plain; charset=utf-8", b"Not found", head)
            return
        try:
            if path.stat().st_size > 25 * 1024 * 1024:
                self._send_bytes(413, "text/plain; charset=utf-8", b"Too large", head)
                return
            body = path.read_bytes()
        except OSError:
            self._send_bytes(404, "text/plain; charset=utf-8", b"Not found", head)
            return
        ctype = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self._send_bytes(200, ctype, body, head, cache="public, max-age=86400")


def _web_value(cfg: Config, key: str, default):
    web = cfg.get("web", None)
    return web.get(key, default) if isinstance(web, dict) else default


def create_server(cfg: Config, db: Optional[Database] = None,
                  host: Optional[str] = None, port: Optional[int] = None,
                  manager=None) -> _WebServer:
    """Создать HTTP-сервер (не запуская его). Порт 0 — свободный порт.

    manager — CameraManager процесса камеры. Если не передан (панель запущена
    отдельно), показываются статусы из data/db/camera_status.json, а кнопки
    управления отключены.
    """
    owns_db = db is None
    if db is None:
        db = Database(cfg)
    bind_host = host if host is not None else str(_web_value(cfg, "host", "127.0.0.1"))
    bind_port = int(port if port is not None else _web_value(cfg, "port", 8088))
    token = str(_web_value(cfg, "token", "") or "")
    title = str(_web_value(cfg, "title", "MLCamera — история событий"))
    control = bool(_web_value(cfg, "control", True))
    events_dir = cfg.storage.resolve(cfg.storage.events_dir).resolve()
    server = _WebServer((bind_host, bind_port), EventsHandler, cfg=cfg, db=db,
                        token=token, title=title, events_dir=events_dir,
                        owns_db=owns_db, manager=manager, control=control)
    return server


def start_background_server(cfg: Config, db: Optional[Database] = None,
                            host: Optional[str] = None,
                            port: Optional[int] = None,
                            manager=None) -> _WebServer:
    """Поднять панель в фоновом потоке (для автозапуска из src/main.py)."""
    server = create_server(cfg, db=db, host=host, port=port, manager=manager)
    thread = threading.Thread(target=server.serve_forever, name="mlcamera-web", daemon=True)
    thread.start()
    bound_host, bound_port = server.server_address[:2]
    shown = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    log.info("Веб-панель истории: http://%s:%d", shown, bound_port)
    if server.manager is not None:
        log.info("Управление камерами: доступно (панель в процессе камеры)")
    else:
        log.info("Управление камерами: только просмотр (панель запущена отдельно)")
    if server.token:
        log.info("Веб-панель защищена токеном: web.token в конфиге")
    return server


def serve_forever(cfg: Config, host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Блокирующий запуск (scripts/web_panel.py)."""
    server = create_server(cfg, host=host, port=port)
    bound_host, bound_port = server.server_address[:2]
    shown = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    print(f"MLCamera веб-панель: http://{shown}:{bound_port}")
    if server.token:
        print(f"Токен: {server.token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
