"""Export a portable, offline, interactive BFS / A* playback using real traces.

Usage:
  python scripts/export_live_demo.py --source 918454 --target 990282
This file is for presentation/demo convenience; the Streamlit Live Search tab
runs arbitrary valid SNAP IDs directly.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roadlab.graph import RoadGraph
from roadlab.landmarks import Landmarks
from roadlab.search import bfs, astar, validate_path
from roadlab.trace import TraceRecorder
from roadlab.live import (FRAMES, HEIGHT, WIDTH, landmark_projection,
                          local_projection, build_live_scene)


def build(source: int, target: int, output: Path):
    graph = RoadGraph.load(ROOT / 'data/graph.npz')
    landmarks = Landmarks.load(ROOT / 'data/landmarks.npz', graph)
    s = graph.internal_id(source)
    t = graph.internal_id(target)
    if s == t or not graph.same_component(s, t):
        raise ValueError('Choose distinct, connected SNAP nodes')
    b_record = TraceRecorder(graph.n, s)
    b = bfs(graph, source, target, trace=b_record)
    a_record = TraceRecorder(graph.n, s)
    a = astar(graph, source, target, landmarks, trace=a_record)
    if not (b.found and a.found and b.hops == a.hops and
            validate_path(graph,b) and validate_path(graph,a)):
        raise AssertionError('Recorded routes did not pass correctness validation')
    bt, at = b_record.snapshot(), a_record.snapshot()
    giant = int(graph.component[s]) == graph.largest_component
    coords = landmark_projection(graph, landmarks) if giant else local_projection(graph,s,t)
    scene = build_live_scene(graph,coords,bt,at,b,a,
                            'ALT graph-distance PCA' if giant else 'component graph-distance PCA')
    def encode(a: np.ndarray) -> str:
        return base64.b64encode(a.astype('<u4').tobytes()).decode('ascii')
    payload = dict(
        src=source, dst=target, hops=b.hops,
        b_total=b.expanded, a_total=a.expanded,
        b_discovered=b.discovered, a_discovered=a.discovered,
        projection=scene.projection_name,
        bounds=[list(scene.x_bounds),list(scene.y_bounds)],
        coords=scene.xy.round(3).tolist(), node_ids=scene.node_ids.tolist(),
        source_xy=scene.source_xy.tolist(), target_xy=scene.target_xy.tolist(),
        b_path=scene.bfs_path_xy.round(3).tolist(),
        a_path=scene.astar_path_xy.round(3).tolist(),
        b_seen=bt.discovered_at[scene.nodes].tolist(),
        a_seen=at.discovered_at[scene.nodes].tolist(),
        b_settled=bt.settled_at[scene.nodes].tolist(),
        a_settled=at.settled_at[scene.nodes].tolist(),
        b_prefix=bt.discovered_prefix.tolist(),
        a_prefix=at.discovered_prefix.tolist(),
        budgets=scene.budgets.tolist(),
        b_heat=encode(scene.bfs_heatmaps),
        a_heat=encode(scene.astar_heatmaps),
        width=WIDTH,height=HEIGHT,frames=FRAMES)
    output.parent.mkdir(parents=True, exist_ok=True)
    html = HTML.replace('/* PLOTLY_SCRIPT */',get_plotlyjs())
    html = html.replace('/* DATA_PAYLOAD */',json.dumps(payload,separators=(',',':')))
    output.write_text(html,encoding='utf-8')
    print('Portable replay exported:',output, 'MB:',round(output.stat().st_size / 1e6,2))


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Texas RoadNet · Actual BFS vs A* Search Playback</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{background:#080F1B;margin:0;color:#E8EFF8;font-family:Inter,ui-sans-serif,system-ui,Arial,sans-serif}
main{max-width:1460px;margin:auto;padding:28px 32px 38px}.eyebrow{font-size:11px;letter-spacing:.19em;font-weight:750;color:#37DCCB}
header{background:linear-gradient(120deg,#182D45 0%,#112133 70%,#0C1E25);border:1px solid #293E55;border-radius:17px;padding:23px 27px 24px}
h1{margin:7px 0 6px;font-weight:750;font-size:33px;letter-spacing:-1.3px}p{color:#A6B8CD;margin:5px 0;font-size:13px;line-height:1.5}
.pill{border:1px solid #3A626E;color:#83F2E3;border-radius:20px;font-size:11px;font-weight:700;padding:5px 11px;display:inline-block;margin:12px 7px 0 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:18px}.stat{background:#101F32;border:1px solid #263950;padding:15px 18px;border-radius:12px}.stat small{font-size:11px;color:#93A8BF}.stat strong{font-size:24px;letter-spacing:-.6px;display:block;margin-top:3px}.stat:nth-child(3) strong{color:#37DCCB}
.controls{background:#101C2C;border:1px solid #263B52;border-radius:14px;margin:17px 0 13px;padding:14px 17px}.controltop{display:flex;gap:11px;flex-wrap:wrap;align-items:center}button{border-radius:8px;padding:10px 19px;font-size:13px;cursor:pointer;border:1px solid #34455F;background:#1C3047;color:#DCEBFA;font-weight:700}button:hover{border-color:#4DE2D2}button.primary{background:#19BAAA;color:#071C22;border-color:#19BAAA}select{background:#192B40;color:#DEE8F6;border:1px solid #35465C;border-radius:8px;padding:10px}.budget{margin-left:auto;color:#8BE5D8;font-weight:700;font-size:12px}input[type=range]{width:100%;accent-color:#32D3C2;cursor:pointer;margin:15px 0 4px}.timelabel{display:flex;justify-content:space-between;font-size:11px;color:#8296B0}
.maps{display:grid;grid-template-columns:1fr 1fr;gap:14px}.panel{border:1px solid #273C55;border-radius:15px;background:#0C1728;overflow:hidden}.heading{padding:17px 20px 6px;display:flex;justify-content:space-between;align-items:center}.heading strong{font-size:19px}.heading span{font-size:12px;color:#96A9C0}.plot{height:440px}.smstats{display:grid;grid-template-columns:repeat(3,1fr);padding:6px 18px 18px;gap:12px}.mini{border-top:1px solid #2D4359;padding-top:9px}.mini small{color:#91A4BC;font-size:11px}.mini strong{font-size:19px;display:block;margin-top:3px;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0 12px;font-size:12px;color:#AAC2D8}.dot{height:9px;width:9px;border-radius:50%;display:inline-block;margin-right:7px}.notes{background:#0E1C2C;border:1px solid #24384F;border-radius:13px;padding:17px 22px;color:#A7BBD3;font-size:12px;line-height:1.75}.notes b{color:#E5F0FF}.result{font-size:12px;color:#60E4CE;margin-top:9px;min-height:20px}
@media(max-width:880px){main{padding:13px}.maps{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:26px}.budget{margin-left:0}.plot{height:390px}}
</style>
<script>/* PLOTLY_SCRIPT */</script></head><body><main>
<header><div class="eyebrow">TEXAS ROADNET / LIVE SEARCH INSTRUMENTATION</div><h1>BFS <span style="color:#42E1CC">vs</span> A* · Search in Motion</h1><p>Original SNAP road network · exact shortest-hop routing · synchronized, real-event replay</p><span class="pill">LIVE EVENT PLAYBACK</span><span class="pill">GRAPH-DISTANCE PROJECTION</span><span class="pill">NO INVENTED GPS</span></header>
<div class="stats"><div class="stat"><small>Route</small><strong id="route"></strong></div><div class="stat"><small>BFS · Settled</small><strong id="btotal"></strong></div><div class="stat"><small>ALT A* · Settled</small><strong id="atotal"></strong></div><div class="stat"><small>Exploration reduction</small><strong id="saving"></strong></div></div>
<div class="controls"><div class="controltop"><button id="reset">↺ Restart</button><button id="play" class="primary">▶ Play</button><button id="finish">⏭ Finish</button><label style="font-size:12px;color:#A8BCD0">Speed <select id="speed"><option value="1">1 fps</option><option value="3">3 fps</option><option value="6" selected>6 fps</option><option value="12">12 fps</option><option value="24">24 fps</option></select></label><div class="budget" id="budget"></div></div><input id="timeline" type="range" min="0" max="120" value="0"><div class="timelabel"><span>INITIAL SEARCH STATE</span><span id="step">FRAME 0 / 120</span><span>ROUTES VERIFIED</span></div><div class="result" id="result"></div></div>
<div class="maps"><section class="panel"><div class="heading"><strong>Breadth-First Search</strong><span>Uninformed · FIFO frontier</span></div><div id="plotB" class="plot"></div><div class="smstats"><div class="mini"><small>Settled nodes</small><strong id="be"></strong></div><div class="mini"><small>Discovered nodes</small><strong id="bd"></strong></div><div class="mini"><small>Frontier nodes</small><strong id="bf"></strong></div></div></section>
<section class="panel"><div class="heading"><strong>A* <span style="color:#3FDDCF">· ALT</span></strong><span>Informed · admissible heuristic</span></div><div id="plotA" class="plot"></div><div class="smstats"><div class="mini"><small>Settled nodes</small><strong id="ae"></strong></div><div class="mini"><small>Discovered nodes</small><strong id="ad"></strong></div><div class="mini"><small>Frontier nodes</small><strong id="af"></strong></div></div></section></div>
<div class="legend"><span><i class="dot" style="background:#37CDBF"></i>Settled / explored</span><span><i class="dot" style="background:#FFC56D"></i>Discovered, not yet settled (frontier)</span><span><i class="dot" style="background:#566780"></i>Unseen (sampled)</span><span><i class="dot" style="background:#F1EC76"></i>Final validated road path</span></div>
<div class="notes"><b>Why the plot is scientifically defensible.</b> This is a <b>topology-derived embedding</b> computed from road-network shortest-hop distances to landmarks. It is <b>not</b> a geographic map of Texas. Both charts use identical node positions, viewport, samples and expansion budgets. Colored density corresponds to <b>all</b> actually settled nodes; individual dots are a reproducible sample (for performance), with exact first-discovery and settlement times. Both search algorithms were run on the complete real graph. The path shown after completion follows actual dataset edges and may differ between equally optimal routes. Search finishes before playback so you can pause, replay, and scrub without running it again. The expansion timeline allocates more frames to early exploration.</div>
</main><script>
const D=/* DATA_PAYLOAD */;
const $=id=>document.getElementById(id);
const fmt=n=>n.toLocaleString('en-US');
$('route').textContent=fmt(D.hops)+' hops';$('btotal').textContent=fmt(D.b_total);$('atotal').textContent=fmt(D.a_total);$('saving').textContent=((1-D.a_total/D.b_total)*100).toFixed(1)+'%';
function decodeBase64(b64){const binary=atob(b64);const arr=new Uint32Array(binary.length/4);for(let i=0;i<arr.length;i++){const j=i*4;arr[i]=(binary.charCodeAt(j)|(binary.charCodeAt(j+1)<<8)|(binary.charCodeAt(j+2)<<16)|(binary.charCodeAt(j+3)<<24))>>>0;}return arr;}
const heatB=decodeBase64(D.b_heat),heatA=decodeBase64(D.a_heat);
const xs=D.coords.map(p=>p[0]),ys=D.coords.map(p=>p[1]);
const hx=Array.from({length:D.width},(_,i)=>D.bounds[0][0]+(D.bounds[0][1]-D.bounds[0][0])*i/(D.width-1));
const hy=Array.from({length:D.height},(_,i)=>D.bounds[1][0]+(D.bounds[1][1]-D.bounds[1][0])*i/(D.height-1));
const chartLayout={paper_bgcolor:'#0C1728',plot_bgcolor:'#0C1728',font:{color:'#B1C5DA',family:'Inter,Arial,sans-serif'},margin:{l:10,r:10,t:8,b:18},xaxis:{visible:false,range:D.bounds[0]},yaxis:{visible:false,range:D.bounds[1]},showlegend:false,hovermode:'closest',uirevision:'fixed'};
const config={displaylogo:false,responsive:true,displayModeBar:false};
function mapHeat(data,k){const w=D.width,h=D.height,start=k*w*h,out=[];for(let y=0;y<h;y++){const row=[];for(let x=0;x<w;x++){const c=data[start+y*w+x];row.push(c?Math.log1p(c):null);}out.push(row);}return out;}
const emptyColors=D.node_ids.map(_=>'rgba(80,96,124,0.20)');
function pin(label,p,color,id){return {type:'scatter',mode:'markers+text',x:[p[0]],y:[p[1]],name:label,marker:{size:18,color:color,line:{color:'#071525',width:2}},text:[label],textposition:'top center',textfont:{color:color,size:16},hovertemplate:'SNAP '+id+'<extra></extra>'};}
function initialPlot(heat,path){return [
 {type:'heatmap',x:hx,y:hy,z:mapHeat(heat,0),colorscale:[[0,'#102A40'],[.22,'#155168'],[.62,'#198F9A'],[1,'#41F1D2']],opacity:.58,showscale:false,hoverinfo:'skip'},
 {type:'scattergl',mode:'lines',x:path.map(p=>p[0]),y:path.map(p=>p[1]),line:{color:'#F1EC76',width:4},visible:false,hoverinfo:'skip'},
 {type:'scattergl',mode:'markers',x:xs,y:ys,customdata:D.node_ids,marker:{color:emptyColors,size:5},hovertemplate:'SNAP %{customdata}<extra></extra>'},
 pin('S',D.source_xy,'#FFFFFF',D.src),pin('T',D.target_xy,'#FF7E98',D.dst)];}
let ready=false,frame=0,playing=false,startedAt=0,playStart=0,updating=false,queuedFrame=null;
Promise.all([Plotly.newPlot('plotB',initialPlot(heatB,D.b_path),chartLayout,config),Plotly.newPlot('plotA',initialPlot(heatA,D.a_path),chartLayout,config)]).then(()=>{ready=true;render(0);});
function stateColors(seen,settled,budget){const colors=[];for(let i=0;i<seen.length;i++){if(settled[i]>=1&&settled[i]<=budget)colors.push('rgba(55,205,191,0.92)');else if(seen[i]>=0&&seen[i]<=budget)colors.push('rgba(255,197,109,0.98)');else colors.push('rgba(80,96,124,0.16)');}return colors;}
async function updatePlot(id,heat,seen,settled,prefix,total,budget,f){const finished=budget>=total;await Plotly.restyle(id,{z:[mapHeat(heat,f)]},[0]);await Plotly.restyle(id,{'marker.color':[stateColors(seen,settled,budget)]},[2]);await Plotly.restyle(id,{visible:finished},[1]);return [Math.min(total,budget),prefix[Math.min(total,budget)]];}
async function render(f){frame=Math.max(0,Math.min(D.frames,Math.round(f)));$('timeline').value=frame;$('step').textContent='FRAME '+frame+' / '+D.frames;const budget=D.budgets[frame];$('budget').textContent=fmt(budget)+' shared expansion budget';
const be=Math.min(D.b_total,budget),ae=Math.min(D.a_total,budget),bd=D.b_prefix[be],ad=D.a_prefix[ae];
$('be').textContent=fmt(be);$('bd').textContent=fmt(bd);$('bf').textContent=fmt(bd-be);
$('ae').textContent=fmt(ae);$('ad').textContent=fmt(ad);$('af').textContent=fmt(ad-ae);
$('result').textContent=ae===D.a_total&&be<D.b_total?'A* has reached the goal; BFS is still exploring at the same node-expansion budget.':ae===D.a_total&&be===D.b_total?'Both searches completed. Yellow lines show the actual verified shortest paths.':'Orange nodes form each algorithm’s frontier; teal density counts settled nodes.';
if(!ready)return;if(updating){queuedFrame=frame;return;}updating=true;const drawnFrame=frame;await Promise.all([updatePlot('plotB',heatB,D.b_seen,D.b_settled,D.b_prefix,D.b_total,budget,frame),updatePlot('plotA',heatA,D.a_seen,D.a_settled,D.a_prefix,D.a_total,budget,frame)]);updating=false;if(queuedFrame!==null){const nextFrame=queuedFrame;queuedFrame=null;if(nextFrame!==drawnFrame)render(nextFrame);}}
function pause(){playing=false;$('play').textContent='▶ Play';}
$('play').onclick=()=>{if(playing){pause();return;}if(frame>=D.frames){frame=0;render(0)}playing=true;playStart=frame;startedAt=performance.now();$('play').textContent='Ⅱ Pause'};
$('reset').onclick=()=>{pause();render(0)};$('finish').onclick=()=>{pause();render(D.frames)};$('timeline').oninput=e=>{pause();render(+e.target.value)};
setInterval(()=>{if(!playing||!ready)return;const nf=Math.min(D.frames,playStart+Math.floor((performance.now()-startedAt)*Number($('speed').value)/1000));if(nf>frame){render(nf);if(nf>=D.frames)pause();}},90);
</script></body></html>'''

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=int,default=918454)
    p.add_argument('--target',type=int,default=990282)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/live_search_demo.html')
    args=p.parse_args()
    build(args.source,args.target,args.output)
