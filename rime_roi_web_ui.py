#!/usr/bin/env python3
"""Browser-based ROI selector for FEAST (headless AutoDL friendly).

Run on AutoDL and visit via SSH port forwarding. Writes an ROI JSON compatible
with `python -m FEAST.enhance_searaft_roi --roi-file ...`.
Uses only Python stdlib and the existing OpenCV installation.
"""
import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FEAST · ROI Selection</title><style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}body{background:#10141c;color:#e7ebf2;margin:0;padding:28px}main{max-width:1120px;margin:auto}h1{font-size:24px;margin:0 0 6px}.muted{color:#a0adbf}.toolbar{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin:15px 0}button{background:#315ee5;border:0;border-radius:8px;color:white;padding:9px 14px;font-size:14px;cursor:pointer}button.secondary{background:#293449}button:disabled{opacity:.5;cursor:not-allowed}.frame{display:inline-block;position:relative;max-width:100%;background:#05080d;border:1px solid #46536a;border-radius:8px;overflow:hidden;touch-action:none;line-height:0}.frame canvas{width:100%;height:auto;display:block;cursor:crosshair;touch-action:none}.layout{display:grid;grid-template-columns:minmax(0,1fr) 240px;gap:18px;align-items:start}aside{border:1px solid #334052;border-radius:9px;padding:14px;background:#171e29}aside ul{padding-left:25px;line-height:1.9}aside li{font-family:ui-monospace,monospace;font-size:12px}small{display:block;margin-top:12px;color:#a0adbf}#status{font-weight:600} @media(max-width:840px){.layout{grid-template-columns:1fr}}code{color:#ced8f9}
</style></head><body><main><h1>FEAST · 齿轮 ROI 框选</h1><div class="muted">在 RGB 第一帧拖动鼠标画框。可添加多个区域，保存后用于局部 SEA-RAFT 光流增强。</div>
<div class="toolbar"><button id="save">保存 ROI 到 AutoDL</button><button id="undo" class="secondary">撤销上一个</button><button id="clear" class="secondary">清空</button><span id="status" class="muted">尚未保存</span></div>
<div class="layout"><div class="frame"><canvas id="c" width="__WIDTH__" height="__HEIGHT__"></canvas></div><aside><strong>选择区域</strong><div class="muted" style="font-size:12px;margin-top:4px">原始 __WIDTH__ × __HEIGHT__ 坐标</div><ul id="boxes"><li>暂无框选</li></ul><small>框住齿轮和可观察到旋转的齿尖。一个区域对应一个局部增强 ROI，尽量不要框住过多无关背景。</small></aside></div>
<div class="muted" style="margin-top:16px">ROI 文件会写入：<code>__OUTPUT__</code></div></main>
<script>
const img = new Image(); img.src='data:image/png;base64,__IMAGE__';
const c=document.getElementById('c'),ctx=c.getContext('2d'), boxes=[],status=document.getElementById('status');
let start=null,drag=null;
function coords(ev){const r=c.getBoundingClientRect();return [Math.max(0,Math.min(c.width,Math.round((ev.clientX-r.left)*c.width/r.width))),Math.max(0,Math.min(c.height,Math.round((ev.clientY-r.top)*c.height/r.height)))];}
function norm(a,b){return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];}
function draw(){if(!img.complete)return;ctx.drawImage(img,0,0,c.width,c.height);for(let i=0;i<boxes.length;i++){rect(boxes[i],i+1,'#20e6a2')}if(drag)rect(drag,'','#ffc857');document.getElementById('boxes').innerHTML=boxes.length?boxes.map((b,i)=>`<li>${i+1}. [${b.join(', ')}]</li>`).join(''):'<li>暂无框选</li>';}
function rect(b,n,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.strokeRect(b[0]+.5,b[1]+.5,b[2]-b[0],b[3]-b[1]);if(n){ctx.fillStyle=color;ctx.fillRect(b[0],Math.max(0,b[1]-23),25,22);ctx.fillStyle='#0d1720';ctx.font='bold 16px sans-serif';ctx.fillText(String(n),b[0]+7,Math.max(17,b[1]-6));}}
img.onload=draw;
c.addEventListener('pointerdown',e=>{start=coords(e);drag=[...start,...start];c.setPointerCapture(e.pointerId);draw()});
c.addEventListener('pointermove',e=>{if(!start)return;drag=norm(start,coords(e));draw()});
c.addEventListener('pointerup',e=>{if(!start)return;const b=norm(start,coords(e));start=null;drag=null;if(b[2]-b[0]>=6&&b[3]-b[1]>=6){boxes.push(b);status.textContent='有未保存的更改'}draw()});
c.addEventListener('pointercancel',()=>{start=null;drag=null;draw()});
document.getElementById('undo').onclick=()=>{boxes.pop();status.textContent='有未保存的更改';draw()};
document.getElementById('clear').onclick=()=>{boxes.length=0;status.textContent='有未保存的更改';draw()};
document.getElementById('save').onclick=async()=>{if(!boxes.length){status.textContent='请至少框选一个区域';return}status.textContent='正在保存...';try{const r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rois:boxes})});const j=await r.json();if(!r.ok)throw Error(j.error||'保存失败');status.textContent='已保存 '+boxes.length+' 个 ROI';}catch(e){status.textContent='保存失败：'+e.message}};
</script></body></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Output ROI JSON')
    p.add_argument('--host', default='127.0.0.1', help='Default: local-only, use SSH forwarding')
    p.add_argument('--port', type=int, default=7868)
    args = p.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        p.error(f'Cannot open video: {args.video}')
    ok, frame = cap.read()
    cap.release()
    if not ok:
        p.error(f'Cannot read first frame: {args.video}')
    h, w = frame.shape[:2]
    ok, png = cv2.imencode('.png', frame)
    if not ok:
        p.error('Cannot encode first frame')
    src = base64.b64encode(png.tobytes()).decode('ascii')
    page = (HTML.replace('__WIDTH__', str(w)).replace('__HEIGHT__', str(h))
            .replace('__IMAGE__', src).replace('__OUTPUT__', args.output.name)).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ('/', '/index.html'):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(page)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(page)

        def do_POST(self):
            if self.path != '/save':
                self.send_error(404)
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not (0 < length <= 4096):
                    raise ValueError('Request is empty or too large')
                payload = json.loads(self.rfile.read(length).decode('utf-8'))
                raw = payload['rois']
                if not isinstance(raw, list) or not (1 <= len(raw) <= 30):
                    raise ValueError('Select 1 to 30 ROIs')
                rois = []
                for item in raw:
                    if not isinstance(item, list) or len(item) != 4 or any(type(v) is not int for v in item):
                        raise ValueError('Invalid ROI coordinates')
                    x1,y1,x2,y2 = item
                    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and x2-x1 >= 6 and y2-y1 >= 6):
                        raise ValueError('ROI outside image or too small')
                    rois.append(item)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                tmp = args.output.with_suffix(args.output.suffix + '.tmp')
                tmp.write_text(json.dumps({'width':w,'height':h,'rois':rois}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                tmp.replace(args.output)
                print(f'Saved {len(rois)} ROI(s): {args.output}', flush=True)
                body = json.dumps({'ok': True, 'count':len(rois)}).encode()
                self.send_response(200)
            except (ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
                body = json.dumps({'error':str(exc)}).encode()
                self.send_response(400)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((args.host,args.port), Handler)
    print(f'ROI UI running: http://{args.host}:{args.port}', flush=True)
    print(f'Video frame: {w}x{h}; saving to {args.output}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
