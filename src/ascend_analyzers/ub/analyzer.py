#!/usr/bin/env python3
"""Analyze BiSheng MemoryDisplay JSON; standard library only, Python >= 3.9."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys


SOURCE_COMMIT = "cb4fc6a96e157953fe63457d7c5f832d086fc0c3"
SOURCE_BASE = f"https://github.com/Ascend/AscendNPU-IR/blob/{SOURCE_COMMIT}/bishengir/lib/Dialect/HIVM/Transforms/"


def integer(value, context):
    if type(value) is not int or value < 0:
        raise ValueError(f"{context}: expected a nonnegative integer, got {value!r}")
    return value


def union_bytes(intervals):
    """Measure a union of half-open BYTE address intervals."""
    total, right = 0, 0
    for lo, hi in sorted(intervals):
        total += max(0, hi - max(lo, right))
        right = max(right, hi)
    return total


def shape_bytes(buffer):
    """Diagnostic only: dense static scalar memref size, never allocation size."""
    match = re.search(r"memref<((?:\d+x)*)(bf16|[fiu]\d+)(?=\s*[,>])", buffer)
    if not match:
        return None
    dims = [int(x) for x in match[1].split("x") if x]
    bits = 16 if match[2] == "bf16" else int(match[2][1:])
    return (math.prod(dims) * bits + 7) // 8


def analyze_record(record, index):
    if not isinstance(record, dict):
        raise ValueError(f"Record[{index}] must be an object")
    if record.get("status") != "success" or record.get("error_info"):
        raise ValueError(f"Record[{index}]: incomplete/failed allocation: {record.get('status')}, {record.get('error_info')}")
    rows = record.get("memory_info_array")
    if not isinstance(rows, list):
        raise ValueError(f"Record[{index}].memory_info_array must be an array")
    groups, seen, warnings, duplicates = {}, set(), [], 0
    for i, row in enumerate(rows):
        ctx = f"Record[{index}].memory_info_array[{i}]"
        if not isinstance(row, dict):
            raise ValueError(f"{ctx} must be an object")
        fingerprint = json.dumps(row, sort_keys=True)
        if fingerprint in seen:
            duplicates += 1
            continue
        seen.add(fingerprint)
        extent = integer(row.get("extent"), f"{ctx}.extent")
        if extent % 8:
            raise ValueError(f"{ctx}.extent is not byte-aligned bits; unsupported schema")
        size = extent // 8
        life, offsets = row.get("life_time_in_ir"), row.get("offset")
        if not isinstance(life, list) or len(life) != 2:
            raise ValueError(f"{ctx}: lifetime must contain [start, end]")
        start, end = [integer(x, f"{ctx}.lifetime") for x in life]
        if end < start:
            raise ValueError(f"{ctx}: reversed lifetime")
        alloc = integer(row.get("alloc_time_in_ir"), f"{ctx}.alloc_time_in_ir")
        if alloc != start:
            warnings.append(f"row {i}: alloc_time {alloc} differs from lifetime start {start}; using lifetime")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(f"{ctx}: missing allocated offsets")
        offsets = [integer(x, f"{ctx}.offset") for x in offsets]
        if len(set(offsets)) != len(offsets):
            warnings.append(f"row {i}: repeated offsets collapsed")
        offsets = sorted(set(offsets))
        tmp = row.get("is_tmpbuf")
        if type(tmp) is not bool:
            raise ValueError(f"{ctx}.is_tmpbuf must be boolean")
        buffer, location = row.get("buffer"), row.get("source_location", "")
        if not isinstance(buffer, str) or not isinstance(location, str):
            raise ValueError(f"{ctx}: buffer/source_location must be strings")
        nominal = shape_bytes(buffer)
        if nominal is not None and nominal != size:
            warnings.append(f"row {i}: memref payload {nominal} B != storage extent {size} B; retaining extent (inplace storage can differ)")
        # Geometry grouping is for display only. All accounting uses address union.
        key = (tuple(offsets), size, start, end, tmp)
        if key not in groups:
            groups[key] = dict(id=f"g{len(groups)}", offsets=offsets, size_bytes=size,
                               start=start, end=end, tmp=tmp, aliases=[])
        groups[key]["aliases"].append(dict(row=i, buffer=buffer, location=location,
                                           nominal_bytes=nominal, alloc_time=alloc))
    groups = list(groups.values())
    instances = [dict(group=g["id"], offset=o, end_offset=o + g["size_bytes"],
                      start=g["start"], end=g["end"], tmp=g["tmp"])
                 for g in groups for o in g["offsets"] if g["size_bytes"]]
    # Closed integer IR lifetimes become [start, end+1) sweep intervals.
    events = {}
    for i, inst in enumerate(instances):
        events.setdefault(inst["start"], [[], []])[0].append(i)
        events.setdefault(inst["end"] + 1, [[], []])[1].append(i)
    times, active, timeline = sorted(events), set(), []
    for j, t in enumerate(times):
        adds, removes = events[t]
        active.difference_update(removes)
        active.update(adds)
        live = [instances[k] for k in sorted(active)]
        spans = [(a["offset"], a["end_offset"]) for a in live]
        normal = union_bytes((a["offset"], a["end_offset"]) for a in live if not a["tmp"])
        tmp = union_bytes((a["offset"], a["end_offset"]) for a in live if a["tmp"])
        occupied = union_bytes(spans)
        high = max((hi for _, hi in spans), default=0)
        timeline.append(dict(start=t, end=times[j + 1] - 1 if j + 1 < len(times) else t,
                             live_bytes=occupied, live_high_water_bytes=high,
                             normal_bytes=normal, tmp_union_bytes=tmp,
                             tmp_incremental_bytes=occupied - normal,
                             holes_below_live_high_water_bytes=high - occupied,
                             active_groups=sorted({a["group"] for a in live})))
    overlaps = []
    for i, a in enumerate(instances):
        for b in instances[i + 1:]:
            t0, t1 = max(a["start"], b["start"]), min(a["end"], b["end"])
            lo, hi = max(a["offset"], b["offset"]), min(a["end_offset"], b["end_offset"])
            if t0 <= t1 and lo < hi:
                overlaps.append(dict(groups=[a["group"], b["group"]], start=t0,
                                     end=t1, offset=lo, end_offset=hi))
    if overlaps:
        warnings.append(f"{len(overlaps)} overlapping instance pairs outside geometry grouping; may be legal inplace reuse, not proof of a compiler error")
    def peak(field):
        value = max((s[field] for s in timeline), default=0)
        return dict(bytes=value, ir_ranges=[[s["start"], s["end"]] for s in timeline if s[field] == value])
    return dict(record_index=index, raw_rows=len(rows), duplicate_rows_removed=duplicates,
                unique_rows=len(seen), storage_groups=len(groups), physical_instances=len(instances),
                groups=groups, instances=instances, timeline=timeline, warnings=warnings,
                overlaps=overlaps,
                address_high_water_bytes=max((a["end_offset"] for a in instances), default=0),
                ever_covered_bytes=union_bytes((a["offset"], a["end_offset"]) for a in instances),
                peak_live=peak("live_bytes"), peak_tmp=peak("tmp_union_bytes"),
                peak_tmp_incremental=peak("tmp_incremental_bytes"))


def analyze_document(document, capacity_bytes=None):
    if not isinstance(document, dict) or not isinstance(document.get("Record"), list):
        raise ValueError("Expected an object with a Record array")
    if capacity_bytes is not None and integer(capacity_bytes, "capacity_bytes") == 0:
        raise ValueError("capacity_bytes must be positive")
    records = []
    for i, record in enumerate(document["Record"]):
        if not isinstance(record, dict):
            raise ValueError(f"Record[{i}] must be an object")
        if not isinstance(record.get("scope"), str):
            raise ValueError(f"Record[{i}].scope must be a string")
        if record["scope"].lower() == "ub":
            records.append(analyze_record(record, i))
    if not records:
        raise ValueError("No UB records found")
    header = document.get("Header", {})
    if not isinstance(header, dict):
        raise ValueError("Header must be an object")
    return dict(schema_version=1, kernel=str(header.get("KernelName", "unknown")),
                semantics=dict(extent_unit="bit", offset_unit="byte", lifetime="closed integer IR indices",
                               multi_offset="all physical slots reserved during reported lifetime",
                               compiler_source_commit=SOURCE_COMMIT,
                               sources=[SOURCE_BASE + "MemoryDisplay.cpp#L133", SOURCE_BASE + "PlanMemory.cpp#L1646", SOURCE_BASE + "PlanMemory.cpp#L2805"],
                               limitation="dump-derived storage footprint; allocator alignment padding and actual runtime timing are not recoverable"),
                capacity_bytes=capacity_bytes, records=records)


def write_outputs(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "timeline.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["record", "ir_start", "ir_end", "live_bytes", "live_high_water_bytes", "normal_bytes", "tmp_union_bytes", "tmp_incremental_bytes", "holes_below_live_high_water_bytes"]
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in report["records"]:
            for s in r["timeline"]:
                writer.writerow([r["record_index"], s["start"], s["end"]] + [s[k] for k in fields[3:]])
    payload = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (directory / "report.html").write_text(HTML.replace("__DATA__", payload), encoding="utf-8")


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UB Memory Analyzer</title>
<style>
body{font:15px system-ui,sans-serif;background:#f3f6fa;color:#182b42;margin:28px auto;max-width:1400px;padding:0 22px}h1{font-size:28px}h2{font-size:20px;margin-top:30px}p{line-height:1.7}.muted{color:#51657c}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{background:white;padding:20px;border-radius:12px;min-width:220px;border:1px solid #dbe3eb}.number{font-size:28px;font-weight:700;margin-top:10px}svg{background:white;border:1px solid #dbe3eb;border-radius:8px;max-width:100%;height:auto}select,input{font:inherit}input[type=range]{width:65%;vertical-align:middle}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;text-align:left;border-bottom:1px solid #dbe3eb;font-size:13px;vertical-align:top}td{overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:12px}.scroll{overflow:auto;max-height:680px}#detail{padding:12px;background:#e7eef8;white-space:pre-wrap}button{padding:7px 12px;cursor:pointer}a{color:#165dba}.orange{color:#c35f11}.blue{color:#246bcc}
</style>
<h1>UB 使用分析 · <span id="kernel"></span></h1>
<p class="muted">BiSheng / Triton 编译器内存规划 · IR 序号时间轴 · 单个 UB scope<br>extent ÷ 8 → bytes；生命周期 [start, end] 含首尾；所有 multi-buffer 地址计入；共享地址按并集计数。</p>
<label>选择记录 <select id="record"></select></label>
<div class="cards" id="cards" style="margin-top:18px"></div>
<p id="counts" class="muted"></p>
<p>地址高水位是 dump 可见地址的最大末端，包含中间空洞；存活占用是当前地址区间并集。未导出的 allocator 对齐尾部、设备保留空间不在统计中，不能据此还原精确硬件预留量。多个 Record 独立分析，不相加。</p>
<h2>占用随 IR 变化</h2><p><span class="blue">蓝色：存活占用</span> · <span class="orange">橙色：tmp 增量</span> · 灰色：存活地址高水位 · 虚线：全局地址高水位。纵轴 KiB。</p>
<div id="curve"></div>
<h2>查看任意 IR 时刻</h2>
<input id="time" type="range" aria-label="IR 时间"><input id="timeNumber" type="number" aria-label="IR 序号" style="width:100px"> <button id="peak">跳到占用峰值</button>
<p id="snapshot"></p><div id="address"></div><p class="muted">上图按 UB 地址排列；细小区间仅为可见性放大到最小 1 像素。下表列出全部存活组。</p>
<div class="scroll"><table><thead><tr><th>存储组 / 别名</th><th>每槽 / 地址 (B)</th><th>生命周期</th><th>类型</th><th>源位置</th></tr></thead><tbody id="active"></tbody></table></div>
<h2>Buffer 生命周期</h2><p class="muted">每行是一个地址槽；相同布局和生命周期的别名合并显示。悬停看详情，点击查看对应起始时刻。橙色为 tmp，单时刻 tmp 仍占一格。</p>
<div id="gantt" class="scroll"></div><pre id="detail">点击生命周期条查看 buffer IR 和源位置。</pre>
<h2>诊断与统计边界</h2><ul id="warnings"></ul>
<p class="muted">全部字段与峰值区间见 analysis.json；时间分段见 timeline.csv。曲线代表编译器规划的生命周期包络；不推断循环迭代、异步执行时延或各 multi-buffer 槽的实际轮转时刻。</p>
<script type="application/json" id="data">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=b=>`${b.toLocaleString()} B (${(b/1024).toFixed(3)} KiB)`;
let R, lo, hi;
$('kernel').textContent=D.kernel;
D.records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`Record ${r.record_index}`;$('record').append(o)});
function svg(w,h,body){return `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" xmlns="http://www.w3.org/2000/svg">${body}</svg>`}
function text(x,y,s){return `<text x="${x}" y="${y}" font-size="12" fill="#51657c">${esc(s)}</text>`}
function line(x1,y1,x2,y2,color='#dbe3eb'){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}"/>`}
function draw(){
 R=D.records[+$('record').value];lo=R.timeline[0]?.start??0;hi=R.timeline.at(-1)?.end??1;
 const cards=[['地址高水位',R.address_high_water_bytes],['最大存活占用',R.peak_live.bytes],['最大 tmp 占用',R.peak_tmp.bytes]];
 $('cards').innerHTML=cards.map(([n,b])=>`<div class="card">${n}<div class="number">${(b/1024).toFixed(3)} KiB</div>${b.toLocaleString()} bytes</div>`).join('');
 $('counts').textContent=`原始 ${R.raw_rows} 条 · 去除完全重复 ${R.duplicate_rows_removed} 条 · ${R.storage_groups} 个存储组 · ${R.physical_instances} 个非空地址槽 · 存活峰值 IR ${JSON.stringify(R.peak_live.ir_ranges)}`;
 $('warnings').innerHTML=R.warnings.map(w=>`<li>${esc(w)}</li>`).join('')||'<li>无异常字段或额外交叠。</li>';
 if(D.capacity_bytes){const li=document.createElement('li');li.textContent=`给定容量 ${fmt(D.capacity_bytes)}；地址高水位占比 ${(R.address_high_water_bytes/D.capacity_bytes*100).toFixed(2)}%`+(R.address_high_water_bytes>D.capacity_bytes?'；已超过给定容量。':'；仅比较 dump 可见 footprint。');$('warnings').append(li)}
 const w=1280,h=300,x=t=>65+(t-lo)/Math.max(1,hi+1-lo)*1180,max=Math.max(1,R.address_high_water_bytes),y=b=>255-b/max*225;
 let body='';for(let k=0;k<=4;k++){const b=max*k/4;body+=line(65,y(b),1245,y(b))+text(5,y(b)+4,(b/1024).toFixed(1))}
 for(let k=0;k<=6;k++){const t=Math.round(lo+(hi-lo)*k/6);body+=text(x(t),282,t)}
 for(const [field,color] of [['live_high_water_bytes','#8a98a9'],['live_bytes','#246bcc'],['tmp_incremental_bytes','#d87920']]){
  const path=R.timeline.map((s,i)=>`${i?'L':'M'}${x(s.start)},${y(s[field])} L${x(s.end+1)},${y(s[field])}`).join(' ');
  body+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="2.5"/>`;
 }
 body+=`<line x1="65" x2="1245" y1="${y(max)}" y2="${y(max)}" stroke="#596979" stroke-dasharray="5 5"/>`;
 $('curve').innerHTML=svg(w,h,body);
 let gantt='';let row=0;const gx=t=>235+(t-lo)/Math.max(1,hi+1-lo)*1010;
 for(const g of R.groups){for(const offset of g.offsets){const yy=34+row++*26;const name=g.aliases.map(a=>a.buffer.split(' =')[0]).join(', ');gantt+=text(8,yy+14,`${g.id} @${offset}`)+`<rect data-group="${g.id}" x="${gx(g.start)}" y="${yy}" width="${Math.max(2,gx(g.end+1)-gx(g.start))}" height="19" rx="3" fill="${g.tmp?'#d87920':'#397ed8'}"><title>${esc(name+'\n'+fmt(g.size_bytes)+' × '+g.offsets.length+' slots\nIR ['+g.start+', '+g.end+']')}</title></rect>`}}
 for(let k=0;k<=6;k++){const t=Math.round(lo+(hi-lo)*k/6);gantt+=text(gx(t),19,t)}
 $('gantt').innerHTML=svg(w,Math.max(65,40+row*26),gantt);
 $('gantt').querySelectorAll('[data-group]').forEach(el=>el.onclick=()=>{const g=R.groups.find(g=>g.id===el.dataset.group);$('detail').textContent=JSON.stringify(g,null,2);setTime(g.start)});
 for(const id of ['time','timeNumber']){$(id).min=lo;$(id).max=hi;$(id).step=1}
 setTime(R.peak_live.ir_ranges[0]?.[0]??lo);
}
function setTime(t){t=Math.max(lo,Math.min(hi,Math.round(Number(t)||lo)));$('time').value=$('timeNumber').value=t;
 const s=R.timeline.find(s=>s.start<=t&&s.end>=t),live=R.groups.filter(g=>g.start<=t&&t<=g.end);
 $('snapshot').textContent=`IR ${t} · 存活 ${fmt(s?.live_bytes??0)} · tmp ${fmt(s?.tmp_union_bytes??0)} · tmp 增量 ${fmt(s?.tmp_incremental_bytes??0)} · 存活高水位 ${fmt(s?.live_high_water_bytes??0)}`;
 $('active').innerHTML=live.map(g=>`<tr><td>${esc(g.id)}<pre>${esc(g.aliases.map(a=>a.buffer).join('\n'))}</pre></td><td>${fmt(g.size_bytes)}<br>${g.offsets.join(', ')}</td><td>[${g.start}, ${g.end}]</td><td>${g.tmp?'tmp':'normal'}</td><td>${g.aliases.map(a=>esc(a.location)).join('<br>')}</td></tr>`).join('');
 const x=o=>35+o/Math.max(1,R.address_high_water_bytes)*1210;let b=line(35,54,1245,54);
 for(const g of live)for(const o of g.offsets){b+=`<rect x="${x(o)}" y="20" width="${Math.max(1,x(o+g.size_bytes)-x(o))}" height="30" fill="${g.tmp?'#d87920':'#397ed8'}" opacity="0.75"><title>${esc(g.id+' ['+o+', '+(o+g.size_bytes)+') '+fmt(g.size_bytes))}</title></rect>`}
 for(let k=0;k<=5;k++){const o=R.address_high_water_bytes*k/5;b+=text(x(o)-15,78,(o/1024).toFixed(1))}b+=text(1120,100,'UB address / KiB');$('address').innerHTML=svg(1280,110,b);
}
$('record').onchange=draw;$('time').oninput=e=>setTime(e.target.value);$('timeNumber').oninput=e=>setTime(e.target.value);$('peak').onclick=()=>setTime(R.peak_live.ir_ranges[0]?.[0]??lo);draw();
</script></html>'''


EXPECTED_FILENAME = "memory_info_aiv.json"


def _resolve_input(raw: Path) -> Path:
    """Accept a file path or a directory containing the expected JSON file."""
    if raw.is_dir():
        candidate = raw / EXPECTED_FILENAME
        if not candidate.is_file():
            raise FileNotFoundError(f"{EXPECTED_FILENAME} not found in {raw}")
        return candidate
    if not raw.is_file():
        raise FileNotFoundError(f"{raw} does not exist")
    if raw.name != EXPECTED_FILENAME:
        raise ValueError(f"expected filename {EXPECTED_FILENAME}, got {raw.name}")
    return raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help=f"path to {EXPECTED_FILENAME} or a directory containing it")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("ub_report"))
    parser.add_argument("--capacity-bytes", type=int, help="optional target UB capacity in bytes (no device default)")
    args = parser.parse_args(argv)
    try:
        input_file = _resolve_input(args.input)
        report = analyze_document(json.loads(input_file.read_text(encoding="utf-8")), args.capacity_bytes)
        write_outputs(report, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Kernel: {report['kernel']}")
    for r in report["records"]:
        print(f"Record {r['record_index']}: {r['raw_rows']} rows -> {r['storage_groups']} storage groups / {r['physical_instances']} slots")
        for label, value in [("Address high water", r["address_high_water_bytes"]), ("Peak live union", r["peak_live"]["bytes"]), ("Peak tmp union", r["peak_tmp"]["bytes"])]:
            print(f"  {label}: {value:,} B / {value / 1024:.3f} KiB")
        print(f"  Peak live IR ranges (inclusive): {r['peak_live']['ir_ranges']}")
        for warning in r["warnings"]:
            print(f"  NOTE: {warning}")
        if args.capacity_bytes:
            high = r['address_high_water_bytes']
            print(f"  Capacity: {high / args.capacity_bytes:.2%}" + (" EXCEEDED" if high > args.capacity_bytes else " (dump footprint only)"))
    print(f"Report: {(args.output_dir / 'report.html').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
