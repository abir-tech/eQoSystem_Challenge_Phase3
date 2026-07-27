const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Team eQoSystem";
pres.title = "eQoSystem — Resilient Microgrid Design on Dirac-3";

const DEEP="065A82", TEAL="1C7293", MID="21295C", INK="13233A", MUTE="5B6B7A",
      LIGHT="EAF1F5", WHITE="FFFFFF", GOLD="E0A458", GREEN="2E86AB", RED="B23A48";
const sh = () => ({ type:"outer", color:"000000", blur:7, offset:3, angle:45, opacity:0.13 });


// decorative power-grid motif (replaces abstract bubbles): faint bus-and-line network
function gridMotif(s, ox, oy, scale, glow) {
  const P = [[0,1.2],[1.1,0.4],[2.3,1.0],[3.4,0.3],[4.3,1.3],[1.7,2.1],[3.0,2.4],[0.8,3.0],[2.2,3.3],[4.0,3.1]];
  const E = [[0,1],[1,2],[2,3],[3,4],[1,5],[2,6],[5,7],[5,8],[6,9],[8,9],[2,5]];
  const pt = (i)=>({x: ox + P[i][0]*scale, y: oy + P[i][1]*scale});
  E.forEach(([a,b])=>{ const A=pt(a), B=pt(b);
    s.addShape(pres.shapes.LINE, {x:A.x+0.09, y:A.y+0.09, w:B.x-A.x, h:B.y-A.y,
      line:{color: glow?TEAL:"3A4A7A", width:1.2, transparency: glow?55:35}});
  });
  P.forEach((_,i)=>{ const A=pt(i);
    s.addShape(pres.shapes.OVAL, {x:A.x, y:A.y, w:0.18, h:0.18,
      fill:{color: (i===5||i===8)? GOLD : (glow?TEAL:"46598F"), transparency: glow?25:15},
      line:{color:MID, width:0.5}});
  });
  // one 'island' ring around the gold hub
  s.addShape(pres.shapes.OVAL, {x: ox+P[5][0]*scale-0.28, y: oy+P[5][1]*scale-0.28, w:0.74, h:0.74,
    fill:{type:"none"}, line:{color:GOLD, width:1.5, dashType:"dash", transparency:30}});
}

function titleBar(s,kicker,title,dark){
  s.addText(kicker.toUpperCase(),{x:0.6,y:0.42,w:12,h:0.3,fontFace:"Arial",fontSize:12,bold:true,color:dark?"9EC4D8":TEAL,charSpacing:3,margin:0});
  s.addText(title,{x:0.6,y:0.72,w:12.1,h:0.85,fontFace:"Cambria",fontSize:30,bold:true,color:dark?WHITE:INK,margin:0});
}
function card(s,x,y,w,h,fill){ s.addShape(pres.shapes.ROUNDED_RECTANGLE,{x,y,w,h,rectRadius:0.08,fill:{color:fill||WHITE},shadow:sh()}); }
function chip(s,x,y,n,color){
  s.addShape(pres.shapes.OVAL,{x,y,w:0.5,h:0.5,fill:{color},shadow:sh()});
  s.addText(String(n),{x,y,w:0.5,h:0.5,align:"center",valign:"middle",fontFace:"Cambria",fontSize:20,bold:true,color:WHITE,margin:0});
}

/* 1 Title */
let s=pres.addSlide(); s.background={color:MID};
gridMotif(s, 9.0, 0.5, 0.95, true);
gridMotif(s, 10.6, 4.3, 0.62, false);
s.addText("QCi GLOBAL INDUSTRY CHALLENGE 2026 · PHASE 3 · ENERGY INFRASTRUCTURE",{x:0.7,y:1.5,w:11.5,h:0.4,fontFace:"Arial",fontSize:13,bold:true,color:"9EC4D8",charSpacing:2,margin:0});
s.addText("eQoSystem",{x:0.7,y:2.0,w:11,h:1.0,fontFace:"Cambria",fontSize:58,bold:true,color:WHITE,margin:0});
s.addText("Cost Optimization in Resilient Power Grids",{x:0.7,y:3.05,w:11.5,h:0.7,fontFace:"Cambria",fontSize:30,color:"CFE1EC",margin:0});
s.addText("Two-stage stochastic microgrid design with Dirac-3-native integer Hamiltonians",{x:0.7,y:3.75,w:11.8,h:0.5,fontFace:"Arial",fontSize:16,italic:true,color:"9EC4D8",margin:0});
[["0","critical outage hours","across 20 + 40 scenarios"],["50","qudits","vs 149 binary qubits"],["1.005×","of MILP optimum","certified near-optimal cost"]].forEach((st,i)=>{
  const x=0.7+i*4.0; card(s,x,4.75,3.7,1.9,"1B2C55");
  s.addText(st[0],{x,y:4.95,w:3.7,h:0.9,align:"center",fontFace:"Cambria",fontSize:40,bold:true,color:GOLD,margin:0});
  s.addText(st[1],{x,y:5.85,w:3.7,h:0.35,align:"center",fontFace:"Arial",fontSize:15,bold:true,color:WHITE,margin:0});
  s.addText(st[2],{x,y:6.18,w:3.7,h:0.35,align:"center",fontFace:"Arial",fontSize:12,color:"9EC4D8",margin:0});
});
s.addNotes("Pitch: grids are trees; cut a branch and everything past it dies. We pre-place solar/battery/diesel so neighborhoods island through failures, solved as three Hamiltonians on Dirac-3. Headline: zero critical outage hours, half the qubits, within 0.5% of classical optimum.");

/* 2 Problem */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The problem","A distribution grid is a tree — one cut goes dark");
card(s,0.6,1.85,6.0,4.9,LIGHT);
s.addText([
 {text:"Power enters at one substation (root) and flows outward through feeders to every bus (junction).",options:{breakLine:true,bullet:true,paraSpaceAfter:10}},
 {text:"Radial operation: no loops. Efficient and safe — but fragile.",options:{breakLine:true,bullet:true,paraSpaceAfter:10}},
 {text:"A single line failure de-energizes everything downstream of the break.",options:{breakLine:true,bullet:true,paraSpaceAfter:10}},
 {text:"Weather-driven outages cost tens of billions of dollars per year.",options:{bullet:true}},
],{x:0.95,y:2.2,w:5.4,h:4.2,fontFace:"Arial",fontSize:16,color:INK,valign:"top"});
card(s,6.95,1.85,5.75,4.9,WHITE);
const nx=9.8, ty=2.35;
function node(x,y,c,lbl){ s.addShape(pres.shapes.OVAL,{x,y,w:0.5,h:0.5,fill:{color:c},line:{color:WHITE,width:1.5}});
  if(lbl) s.addText(lbl,{x:x-0.2,y:y+0.5,w:0.9,h:0.25,align:"center",fontSize:9,color:MUTE,margin:0}); }
function link(x1,y1,x2,y2,c,dash){ s.addShape(pres.shapes.LINE,{x:x1,y:y1,w:x2-x1,h:y2-y1,line:{color:c||TEAL,width:2.5,dashType:dash||"solid"}}); }
node(nx,ty,DEEP,"substation"); link(nx+0.25,ty+0.5,nx+0.25,ty+0.9,TEAL); node(nx,ty+0.9,TEAL);
link(nx+0.25,ty+1.4,nx-1.35,ty+1.9,RED); link(nx+0.25,ty+1.4,nx+1.6,ty+1.9,TEAL);
s.addText("✕",{x:nx-0.75,y:ty+1.42,w:0.4,h:0.4,fontSize:18,bold:true,color:RED,align:"center",margin:0});
node(nx-1.6,ty+1.9,"B9C4CC"); link(nx-1.35,ty+2.4,nx-2.1,ty+2.9,"B9C4CC"); link(nx-1.35,ty+2.4,nx-0.7,ty+2.9,"B9C4CC");
node(nx-2.35,ty+2.9,"B9C4CC","dark"); node(nx-0.95,ty+2.9,"B9C4CC","dark");
node(nx+1.35,ty+1.9,TEAL); link(nx+1.6,ty+2.4,nx+0.9,ty+2.9,TEAL); link(nx+1.6,ty+2.4,nx+2.3,ty+2.9,TEAL);
node(nx+0.65,ty+2.9,TEAL,"served"); node(nx+2.05,ty+2.9,TEAL,"served");
s.addText("One failed line (✕) blacks out its entire downstream subtree.",{x:7.15,y:6.25,w:5.3,h:0.4,fontSize:12,italic:true,color:MUTE,margin:0});
s.addNotes("Radial = tree. Red X = failed line; grey nodes go dark. This fragility is the whole motivation.");

/* 3 Idea */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The idea","Islanding: neighborhoods that survive on their own");
[["Microgrid","A grid section with local generation + switches so it can run detached.",DEEP,"MG"],
 ["DER","Distributed Energy Resources: solar (PV), battery (BESS), diesel (DG). Each has different 'firm' output you can rely on when islanded.",TEAL,"☼"],
 ["Tie switch","A normally-open spare link, closed remotely after a failure to reroute power around the break.",GREEN,"⇄"],
 ["Islanding","Deliberately disconnecting a section and running it on its own DERs until repair.",GOLD,"◈"]].forEach((d,i)=>{
  const x=0.6+(i%2)*6.15, y=1.9+Math.floor(i/2)*2.45; card(s,x,y,5.85,2.2,i%2?WHITE:LIGHT);
  s.addShape(pres.shapes.OVAL,{x:x+0.3,y:y+0.35,w:0.7,h:0.7,fill:{color:d[2]},shadow:sh()});
  s.addText(d[3],{x:x+0.3,y:y+0.35,w:0.7,h:0.7,align:"center",valign:"middle",fontSize:18,bold:true,color:WHITE,margin:0});
  s.addText(d[0],{x:x+1.2,y:y+0.3,w:4.4,h:0.5,fontFace:"Cambria",fontSize:20,bold:true,color:INK,margin:0});
  s.addText(d[1],{x:x+1.2,y:y+0.82,w:4.45,h:1.25,fontFace:"Arial",fontSize:13.5,color:INK,valign:"top",margin:0});
});
s.addNotes("Four terms to own. Firmness is the subtlety: diesel 100%, battery ~50% & energy-limited, solar ~25% & useless at night.");

/* 4 Pipeline */
s=pres.addSlide(); s.background={color:MID};
titleBar(s,"The pipeline","Three coupled stages, one Hamiltonian family",true);
[["1","DESIGN","Invest once","Which islands to build, how many PV/BESS/DG units, which tie switches.","50 qudits · degree 3",DEEP],
 ["2","ISLANDING","Per contingency","For each failure: which built islands energize, which ties close.","≤9 qudits · degree 2",TEAL],
 ["3","DISPATCH","Per island · per hour","Generator setpoints — critical load first, then customers.","2–4 qudits · degree 3",GREEN]].forEach((st,i)=>{
  const x=0.6+i*4.15; card(s,x,2.0,3.85,3.5,"1B2C55"); chip(s,x+0.35,2.3,st[0],st[5]);
  s.addText(st[1],{x:x+1.05,y:2.32,w:2.7,h:0.45,fontFace:"Cambria",fontSize:20,bold:true,color:WHITE,margin:0});
  s.addText(st[2],{x:x+1.05,y:2.78,w:2.7,h:0.3,fontFace:"Arial",fontSize:12,italic:true,color:GOLD,margin:0});
  s.addText(st[3],{x:x+0.35,y:3.35,w:3.2,h:1.35,fontFace:"Arial",fontSize:13.5,color:"CFE1EC",valign:"top",margin:0});
  s.addText(st[4],{x:x+0.35,y:4.9,w:3.2,h:0.4,fontFace:"Arial",fontSize:12,bold:true,color:"9EC4D8",margin:0});
  if(i<2) s.addShape(pres.shapes.LINE,{x:x+3.9,y:3.7,w:0.2,h:0,line:{color:GOLD,width:3,endArrowType:"triangle"}});
});
s.addText("Uncertainty (which line fails, when, how long, load & sun) is sampled by Latin-Hypercube into 20 scenarios. Invest before knowing the disaster; evaluate across all of them — a two-stage stochastic problem.",{x:0.6,y:5.75,w:12.1,h:0.9,fontFace:"Arial",fontSize:14.5,color:"CFE1EC",valign:"top",margin:0});
s.addNotes("Spine of the talk: all three stages are the same object — a bounded-integer polynomial — so one machine solves all of them.");

/* 5 Hamiltonian / machine */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The machine","Hamiltonians on an entropy quantum computer");
card(s,0.6,1.9,6.0,4.85,LIGHT);
s.addText("What we hand the machine",{x:0.95,y:2.15,w:5.4,h:0.4,fontFace:"Cambria",fontSize:18,bold:true,color:DEEP,margin:0});
s.addText([
 {text:"A Hamiltonian = a polynomial whose minimum encodes the best decision. 'Solving' = finding the values with lowest value.",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"Dirac-3 physically relaxes photonic states toward that minimum — photon shot noise is the fuel, not the enemy.",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"Native integer 'qudits' (0…d), not just 0/1 qubits.",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"Native polynomials up to degree 5 — no auxiliary variables.",options:{bullet:true}},
],{x:0.95,y:2.6,w:5.35,h:4.0,fontFace:"Arial",fontSize:14.5,color:INK,valign:"top"});
card(s,6.95,1.9,5.75,4.85,WHITE);
s.addText("Why it fits THIS problem",{x:7.3,y:2.15,w:5.0,h:0.4,fontFace:"Cambria",fontSize:18,bold:true,color:TEAL,margin:0});
[["Unit counts (0–8 PV, 0–6 DG…)","one qudit each, not 3 qubits"],
 ["Gated capacity  b·(cap−dem)²","native degree-3 term"],
 ["Diesel fuel curve","native cubic term"]].forEach((r,i)=>{
  const y=2.65+i*1.15; card(s,7.3,y,5.05,0.95,LIGHT);
  s.addText(r[0],{x:7.5,y:y+0.14,w:4.7,h:0.4,fontFace:"Arial",fontSize:14,bold:true,color:INK,margin:0});
  s.addText(r[1],{x:7.5,y:y+0.52,w:4.7,h:0.35,fontFace:"Arial",fontSize:12.5,italic:true,color:TEAL,margin:0});
});
s.addText("On a qubit machine these need binary expansion + quadratization — more variables, more noise.",{x:7.3,y:6.18,w:5.05,h:0.5,fontFace:"Arial",fontSize:12,italic:true,color:MUTE,valign:"top",margin:0});
s.addNotes("Hamiltonian = cost landscape; solving = lowest point. The three 'fit' rows are the quantum-rationale in miniature.");

/* 6 Simulated annealing */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The classical engine","Simulated annealing: how we search the landscape");
card(s,0.6,1.9,6.2,4.85,LIGHT);
s.addText([
 {text:"A blindfolded hiker seeking the lowest valley.",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"'Always downhill' gets stuck in the first dip (local minimum).",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"Annealing starts 'hot': sometimes steps uphill to escape dips. Then cools — uphill moves get rarer — until it settles in a deep valley.",options:{breakLine:true,bullet:true,paraSpaceAfter:9}},
 {text:"Name from metallurgy: cool metal slowly and atoms find a low-energy crystal.",options:{bullet:true}},
],{x:0.95,y:2.2,w:5.55,h:4.2,fontFace:"Arial",fontSize:15,color:INK,valign:"top"});
card(s,7.15,1.9,5.55,4.85,WHITE);
s.addText("Under the hood",{x:7.5,y:2.15,w:4.8,h:0.4,fontFace:"Cambria",fontSize:18,bold:true,color:DEEP,margin:0});
s.addText([
 {text:"Accept a worse move with probability exp(−ΔE / T).",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Temperature T cools geometrically 2.5 → 0.001.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"6 restarts · greedy warm start · ±1 polish.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Delta evaluation: only re-score terms touching the moved variable → 5–15× faster.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Algorithmic twin of what Dirac-3 does physically — and our fair yardstick in E5.",options:{bullet:true}},
],{x:7.5,y:2.6,w:4.9,h:4.0,fontFace:"Arial",fontSize:14,color:INK,valign:"top"});
s.addNotes("Two roles: runs-anywhere classical solver for judges, and the honest baseline in E5 run identically on both encodings.");

/* 7 What Phase 3 asked */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The brief","What Phase 3 required — and where we stand");
[["Three-stage formulation","Design → islanding → dispatch as Hamiltonians","Done"],
 ["Run on Dirac-3 via qBraid","Verified submission format; token-gated run","Ready"],
 ["Reproducible by judges","One command, fixed seeds, <1 min","Done"],
 ["Metrics M1 / M2 / M3","Computed exactly as defined","Done"],
 ["Classical baseline","Exact MILP + brute-force certification","Done"],
 ["Robust under stress","M2 = 0 even at load \u00d71.5 and 20 h outages","Done"]].forEach((r,i)=>{
  const x=0.6+(i%2)*6.15, y=1.9+Math.floor(i/2)*1.63; card(s,x,y,5.85,1.45,i%2?WHITE:LIGHT);
  const ok=r[2]==="Done";
  s.addShape(pres.shapes.OVAL,{x:x+0.3,y:y+0.45,w:0.55,h:0.55,fill:{color:ok?GREEN:GOLD},shadow:sh()});
  s.addText(ok?"✓":"→",{x:x+0.3,y:y+0.45,w:0.55,h:0.55,align:"center",valign:"middle",fontSize:18,bold:true,color:WHITE,margin:0});
  s.addText(r[0],{x:x+1.05,y:y+0.25,w:4.6,h:0.45,fontFace:"Cambria",fontSize:17,bold:true,color:INK,margin:0});
  s.addText(r[1],{x:x+1.05,y:y+0.74,w:4.6,h:0.55,fontFace:"Arial",fontSize:12.5,color:MUTE,valign:"top",margin:0});
  s.addText(r[2].toUpperCase(),{x:x+4.5,y:y+0.25,w:1.15,h:0.35,align:"right",fontFace:"Arial",fontSize:11,bold:true,color:ok?GREEN:GOLD,margin:0});
});
s.addNotes("The single 'Ready' is the hardware run — needs the team qBraid token. Everything else complete and reproducible.");

/* 8 Rebuild */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"The rebuild","From an inherited prototype to a working solution");
card(s,0.6,1.9,6.0,4.85,"F6E9EA");
s.addText("Phase 2 prototype — what we found",{x:0.95,y:2.15,w:5.3,h:0.4,fontFace:"Cambria",fontSize:17,bold:true,color:RED,margin:0});
s.addText([
 {text:"The quantum path NEVER ran — eqc-models calls crash; all solving was classical SciPy.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Islanding degenerate: every island always ON, in every scenario — contingencies ignored.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Candidates disjoint (no shared buses) \u2192 no coupling, so no real optimization.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Coefficient range ≈ 80 dB — unusable on analog hardware.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Reported results NOT producible by the code (no baseline, no time loop, crashes first).",options:{bullet:true}},
],{x:0.95,y:2.6,w:5.35,h:4.0,fontFace:"Arial",fontSize:13.5,color:INK,valign:"top"});
card(s,6.95,1.9,5.75,4.85,"E7F1EC");
s.addText("Phase 3 rebuild — what changed",{x:7.3,y:2.15,w:5.1,h:0.4,fontFace:"Cambria",fontSize:17,bold:true,color:"1E7A4D",margin:0});
s.addText([
 {text:"Verified Dirac-3 converter, bit-exact vs eqc-models; no silent fallback.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Genuine coupled islanding QUBO; failures truly de-energize.",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Gated degree-3 capacity + cubic fuel; dual DER hubs (M2 → 0).",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Coefficients conditioned to 19–57 dB (E3 proves why).",options:{breakLine:true,bullet:true,paraSpaceAfter:8}},
 {text:"Every number regenerates in <1 min, with baselines + certification.",options:{bullet:true}},
],{x:7.3,y:2.6,w:5.1,h:4.0,fontFace:"Arial",fontSize:13.5,color:INK,valign:"top"});
s.addNotes("Matter-of-fact, not disparaging — we inherited it. Point: the current submission is real, reproducible, honest.");

/* 9 Results M1 M2 */
s=pres.addSlide(); s.background={color:WHITE};
titleBar(s,"Results","Customers stay powered — critical loads never drop");
s.addImage({path:"/home/claude/phase3/deck/panel_m1.png",x:0.5,y:1.95,w:6.0,h:4.07});
s.addImage({path:"/home/claude/phase3/deck/panel_m2.png",x:6.75,y:1.95,w:6.0,h:4.07});
s.addText("M1 — worst-hour unserved: 15.8% vs 100% baseline",{x:0.5,y:6.05,w:6.0,h:0.35,align:"center",fontFace:"Arial",fontSize:13,bold:true,color:DEEP,margin:0});
s.addText("M2 — critical-infra hours: 0 vs 174 baseline",{x:6.75,y:6.05,w:6.0,h:0.35,align:"center",fontFace:"Arial",fontSize:13,bold:true,color:GREEN,margin:0});
s.addText("The flat blue line is the headline: critical infrastructure loses zero hours in every scenario — an arithmetic consequence of the 1.25× sizing margin, held under 40 out-of-sample scenarios. Every island-hour also passes LinDistFlow voltage/thermal validation (Vmin 0.986 pu) with hourly battery state-of-charge simulated.",{x:0.6,y:6.5,w:12.1,h:0.7,fontFace:"Arial",fontSize:13,italic:true,color:MUTE,valign:"top",margin:0});
s.addNotes("Pre-empt 'zero looks fake': it's designed and proven, not tuned. Baseline reaches 40 h; ours is zero because we spent M3 to buy it down.");

/* 10 Advantage */
s=pres.addSlide(); s.background={color:MID};
titleBar(s,"Quantum rationale","Measured, not asserted — native encoding pays off",true);
s.addImage({path:"/home/claude/phase3/deck/panel_e5.png",x:0.5,y:2.0,w:6.0,h:4.07});
card(s,6.9,2.0,5.85,4.05,"1B2C55");
s.addText("E5 — same problem, two encodings",{x:7.2,y:2.25,w:5.2,h:0.4,fontFace:"Cambria",fontSize:18,bold:true,color:WHITE,margin:0});
[["Variables","50 → 149","3.0×"],["Poly. terms","196 → 1,238","6.3×"],["SA wall-clock","1× → 4×","4.0×"]].forEach((r,i)=>{
  const y=2.8+i*0.78;
  s.addText(r[0],{x:7.2,y,w:2.3,h:0.5,fontFace:"Arial",fontSize:14,bold:true,color:"CFE1EC",valign:"middle",margin:0});
  s.addText(r[1],{x:9.4,y,w:1.9,h:0.5,fontFace:"Arial",fontSize:13,color:"9EC4D8",valign:"middle",margin:0});
  s.addText(r[2],{x:11.3,y,w:1.2,h:0.5,align:"right",fontFace:"Cambria",fontSize:18,bold:true,color:GOLD,valign:"middle",margin:0});
});
s.addText("Native qudits reach within 1% of the best solution at equal budget; the binary compilation never does. Degree-3 terms would also need quadratization on qubit hardware — worsening analog noise (E3).",{x:7.2,y:5.15,w:5.25,h:0.8,fontFace:"Arial",fontSize:12.5,italic:true,color:"CFE1EC",valign:"top",margin:0});
s.addText("We claim certified parity with classical MILP (1.005×) and a measured encoding advantage — not a fake speedup.",{x:0.6,y:6.35,w:12.1,h:0.6,align:"center",fontFace:"Arial",fontSize:14,bold:true,color:GOLD,margin:0});
s.addNotes("Rubric: 'clear advantage OR rationale'. At 33-bus nobody honestly beats HiGHS; credible story is representational economy, which E5 measures. Say parity out loud.");

/* 12 Close */
s=pres.addSlide(); s.background={color:MID};
gridMotif(s, 10.3, 0.35, 0.7, true);
gridMotif(s, -0.4, 5.0, 0.55, false);
s.addText("WHERE WE ARE \u00b7 WHAT\u2019S NEXT",{x:0.7,y:1.4,w:11,h:0.4,fontFace:"Arial",fontSize:13,bold:true,color:"9EC4D8",charSpacing:2,margin:0});
s.addText("eQoSystem",{x:0.7,y:1.85,w:11,h:0.9,fontFace:"Cambria",fontSize:44,bold:true,color:WHITE,margin:0});
card(s,0.7,3.0,9.15,1.5,"10203F");
s.addText([
 {text:"pip install -e .",options:{breakLine:true,color:"7FE3C0",fontFace:"Courier New"}},
 {text:"eqosystem-experiments --backend sa       # runs anywhere",options:{breakLine:true,color:"CFE1EC",fontFace:"Courier New"}},
 {text:"eqosystem-experiments --backend dirac3   # on qBraid, with token",options:{color:"CFE1EC",fontFace:"Courier New"}},
],{x:0.95,y:3.22,w:8.6,h:1.1,fontSize:13.5,valign:"top"});
[["0 h","critical outage"],["20/20","certified optimal"],["1.005\u00d7","of MILP"],["50","qudits"]].forEach((c,i)=>{
  const x=0.7+i*2.35;
  s.addText(c[0],{x,y:4.85,w:2.2,h:0.6,fontFace:"Cambria",fontSize:26,bold:true,color:GOLD,margin:0});
  s.addText(c[1],{x,y:5.45,w:2.2,h:0.3,fontFace:"Arial",fontSize:11,color:"CFE1EC",margin:0});
});
card(s,10.15,3.0,2.55,2.85,"1B2C55");
s.addText("NEXT STEPS",{x:10.4,y:3.2,w:2.1,h:0.3,fontFace:"Arial",fontSize:11,bold:true,color:GOLD,charSpacing:2,margin:0});
s.addText([
 {text:"Write the 5-page report (draft ready, \u00a75 awaits hardware numbers).",options:{breakLine:true,bullet:true,paraSpaceAfter:7}},
 {text:"Run on Dirac-3 once organizers grant access \u2014 code is submission-ready, one command.",options:{breakLine:true,bullet:true,paraSpaceAfter:7}},
 {text:"Package + submit on Aqora.",options:{bullet:true}},
],{x:10.35,y:3.5,w:2.25,h:2.25,fontFace:"Arial",fontSize:10.5,color:"CFE1EC",valign:"top"});
s.addText("Team eQoSystem \u00b7 ESI-SBA, Algeria \u00b7 QCi Global Industry Challenge 2026 \u2014 Phase 3",{x:0.7,y:6.7,w:11.8,h:0.4,fontFace:"Arial",fontSize:13,italic:true,color:"9EC4D8",margin:0});
s.addNotes("Close on reproducibility + four numbers. Next steps: report writing, Dirac-3 run pending organizer access, Aqora submission.");

pres.writeFile({ fileName:"/home/claude/phase3/deck/eQoSystem_Pipeline.pptx" }).then(()=>console.log("done"));
