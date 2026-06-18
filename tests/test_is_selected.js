// Mini DOM mock: just enough to run the _is_selected JS without jsdom.
// We mock: Element, querySelectorAll, querySelector, textContent, className,
// getAttribute, closest, parentElement, childNodes, getBoundingClientRect.

class El {
  constructor(tag, opts = {}) {
    this.tagName = tag.toUpperCase();
    this._classes = (opts.cls || '').split(/\s+/).filter(Boolean);
    this._attrs = opts.attrs || {};
    this._children = [];
    this.parentElement = null;
    this._textOwn = opts.text || '';
    this._checked = !!opts.checked;
  }
  get className() {
    return { toString: () => this._classes.join(' ') };
  }
  get textContent() {
    return this._textOwn + this._children.map(c => c.textContent).join('');
  }
  get childNodes() {
    if (this._textOwn) {
      return [{ nodeType: 3, textContent: this._textOwn }, ...this._children];
    }
    return this._children;
  }
  get checked() { return this._checked; }
  appendChild(c) { c.parentElement = this; this._children.push(c); return c; }
  getAttribute(k) { return this._attrs[k] || null; }
  closest(sel) {
    // Only support '.classname' selectors here.
    if (!sel.startsWith('.')) return null;
    const want = sel.slice(1);
    let cur = this;
    while (cur) {
      if (cur._classes.includes(want)) return cur;
      cur = cur.parentElement;
    }
    return null;
  }
  querySelector(sel) { return (this.querySelectorAll(sel) || [])[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    const cls = sel.startsWith('.') ? sel.slice(1) : null;
    const tag = !cls ? sel.toUpperCase() : null;
    const walk = (n) => {
      if (n !== this) {
        if (cls && n._classes && n._classes.includes(cls)) out.push(n);
        if (tag && n.tagName === tag) out.push(n);
      }
      (n._children || []).forEach(walk);
    };
    walk(this);
    return out;
  }
  scrollIntoView() {}
  getBoundingClientRect() { return { x: 0, y: 0, left: 0, top: 0, width: 100, height: 30 }; }
}

// Build synthetic DOM matching Styx's structure:
// body > [for each coin]:
//   div.wallet-currency-toggler (outer; display:contents in real life)
//     div.wallet-currency-toggler__title.wct-with-icon
//       span.wct-coin-name "BNB (BEP20)"
const root = new El('body');

function makeTile(coinName, selected = false) {
  const outerCls = 'wallet-currency-toggler' + (selected ? ' active' : '');
  const outer = new El('div', { cls: outerCls });
  const title = new El('div', { cls: 'wallet-currency-toggler__title wct-with-icon' });
  const name  = new El('span', { cls: 'wct-coin-name', text: coinName });
  outer.appendChild(title);
  title.appendChild(name);
  root.appendChild(outer);
  return outer;
}

// Scenario 1: TRX is selected, we check BNB.
makeTile('TRX (TRC20)', true);
makeTile('BNB (BEP20)', false);
makeTile('Bitcoin', false);
makeTile('Ethereum (ERC20)', false);

global.document = {
  querySelectorAll: (sel) => root.querySelectorAll(sel),
  querySelector:    (sel) => root.querySelector(sel),
};
global.getComputedStyle = () => ({ cursor: 'pointer', visibility: 'visible', display: 'block', opacity: '1' });

// Paste the _is_selected JS body (as written in styx_register.py)
const isSelected = (label) => {
  const norm = s => (s || '').replace(/\s+/g, ' ').trim();
  const want = norm(label);
  const nameEls = Array.from(document.querySelectorAll('.wct-coin-name'));

  const ancestorsOf = (el) => {
    let inner = null, outer = null;
    let cur = el;
    for (let i = 0; i < 12 && cur; i++) {
      const cls = (cur.className && cur.className.toString)
                     ? cur.className.toString() : '';
      if (!inner && /wallet-currency-toggler__title/.test(cls)) inner = cur;
      if (!outer && /wallet-currency-toggler(?!__)/.test(cls)) outer = cur;
      if (inner && outer) break;
      cur = cur.parentElement;
    }
    return { inner, outer };
  };
  const stateClassRx = /\bactive\b|\bselected\b|\bchecked\b|\bchosen\b|\bcurrent\b|--is-active|--is-selected|is-active|is-selected/i;
  const isStateOn = (el) => {
    if (!el) return false;
    const cls = (el.className && el.className.toString)
                   ? el.className.toString() : '';
    if (stateClassRx.test(cls)) return true;
    if (el.getAttribute && (
           el.getAttribute('aria-selected') === 'true'
        || el.getAttribute('aria-pressed') === 'true'
        || el.getAttribute('aria-checked') === 'true'
        || el.getAttribute('data-active') === 'true'
        || el.getAttribute('data-selected') === 'true')) return true;
    return false;
  };

  const targetName = nameEls.find(el => norm(el.textContent) === want);
  if (!targetName) {
    const tiles = nameEls.map(n => norm(n.textContent));
    return { ok: false, reason: 'tile not in DOM', tilesFound: tiles };
  }
  const { inner: tInner, outer: tOuter } = ancestorsOf(targetName);
  if (!tInner && !tOuter) return { ok: false, reason: 'no toggler ancestor' };

  let allOuter = Array.from(document.querySelectorAll('.wallet-currency-toggler'))
      .filter(el => !/__/.test(el.className.toString()));
  let allInner = Array.from(document.querySelectorAll('.wallet-currency-toggler__title'));
  const tilesFound = (allOuter.length ? allOuter : allInner).map(t => {
    const n = t.querySelector('.wct-coin-name')
          || (t.parentElement && t.parentElement.querySelector('.wct-coin-name'));
    return n ? norm(n.textContent) : '';
  });

  const directSelected = isStateOn(tInner) || isStateOn(tOuter);

  let inputChecked = false;
  const checkInputs = (el) => {
    if (!el) return;
    el.querySelectorAll('input').forEach(inp => { if (inp.checked) inputChecked = true; });
  };
  checkInputs(tInner);
  checkInputs(tOuter);

  const tilesForDiff = (allOuter.length ? allOuter : allInner);
  let diffTarget;
  if (allOuter.length) diffTarget = tOuter || tInner;
  else                 diffTarget = tInner || tOuter;

  let diffSelected = false;
  let uniqueToTarget = [];
  let uniqueToOthers = [];
  if (diffTarget && tilesForDiff.length > 1) {
    const tClsSet = new Set(diffTarget.className.toString().split(/\s+/).filter(Boolean));
    const sibSets = tilesForDiff.filter(t => t !== diffTarget)
                                .map(t => new Set(t.className.toString().split(/\s+/).filter(Boolean)));
    for (const c of tClsSet) if (sibSets.every(s => !s.has(c))) uniqueToTarget.push(c);
    if (sibSets.length > 0) {
      const inter = new Set(sibSets[0]);
      for (let i = 1; i < sibSets.length; i++) {
        for (const c of Array.from(inter)) if (!sibSets[i].has(c)) inter.delete(c);
      }
      for (const c of inter) if (!tClsSet.has(c)) uniqueToOthers.push(c);
    }
    diffSelected = uniqueToTarget.length > 0 || uniqueToOthers.length > 0;
  }

  let highlighted = null;
  const tilesForHL = allOuter.length ? allOuter : allInner;
  for (const t of tilesForHL) {
    let on = isStateOn(t);
    if (!on) {
      if (allOuter.length) {
        const innerOf = t.querySelector('.wallet-currency-toggler__title');
        if (isStateOn(innerOf)) on = true;
      } else {
        const outerOf = t.closest('.wallet-currency-toggler');
        if (isStateOn(outerOf)) on = true;
      }
    }
    if (on) {
      const n = t.querySelector('.wct-coin-name')
            || (t.parentElement && t.parentElement.querySelector('.wct-coin-name'));
      highlighted = n ? norm(n.textContent) : '';
      break;
    }
  }

  return {
    ok: true, selected: directSelected || inputChecked || diffSelected,
    directSelected, inputChecked, diffSelected,
    uniqueToTarget, uniqueToOthers, highlighted,
    innerCls: tInner ? tInner.className.toString() : null,
    outerCls: tOuter ? tOuter.className.toString() : null,
    tilesFound,
  };
};

console.log('--- Scenario 1: TRX selected, checking BNB ---');
const r1 = isSelected('BNB (BEP20)');
console.log(JSON.stringify(r1, null, 2));
console.assert(r1.selected === false, 'BNB should NOT be selected when TRX is');
console.assert(r1.highlighted === 'TRX (TRC20)', 'highlighted should be TRX');
console.log('PASS');

// Scenario 2: BNB selected, checking BNB
console.log('\n--- Scenario 2: now make BNB selected, recheck BNB ---');
const tiles = root._children;
tiles[0]._classes = ['wallet-currency-toggler'];  // TRX off
tiles[1]._classes = ['wallet-currency-toggler', 'active'];  // BNB on
const r2 = isSelected('BNB (BEP20)');
console.log(JSON.stringify(r2, null, 2));
console.assert(r2.selected === true, 'BNB should be selected');
console.assert(r2.highlighted === 'BNB (BEP20)', 'highlighted should be BNB');
console.log('PASS');

// Scenario 3: Selected class on the INNER __title (outer has nothing)
console.log('\n--- Scenario 3: selected class on inner __title, outer clean ---');
// Reset outer
tiles.forEach((t, i) => t._classes = ['wallet-currency-toggler']);
// Put 'active' on BNB's inner __title
const bnbInner = tiles[1]._children[0];
bnbInner._classes = ['wallet-currency-toggler__title', 'wct-with-icon', 'active'];
const r3 = isSelected('BNB (BEP20)');
console.log(JSON.stringify(r3, null, 2));
console.assert(r3.directSelected === true, 'inner active should be picked up');
console.assert(r3.highlighted === 'BNB (BEP20)', 'highlighted via inner should be BNB');
console.log('PASS');

console.log('\nAll scenarios PASSED.');
