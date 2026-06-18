// Synthetic test of the "find product row + cart icon" JS heuristic in
// do_buy_product(). Mocks just enough of the DOM and getBoundingClientRect
// to validate that:
//   - The Firstmail row is correctly identified by the truncated label.
//   - The cart-icon inside that row is correctly chosen (not the mail icon).
class El {
  constructor(tag, opts = {}) {
    this.tagName = tag.toUpperCase();
    this._classes = (opts.cls || '').split(/\s+/).filter(Boolean);
    this._attrs = opts.attrs || {};
    this._children = [];
    this.parentElement = null;
    this._textOwn = opts.text || '';
    this._rect = opts.rect || null;
    this.onclick = opts.onclick || null;
  }
  get className() { return { toString: () => this._classes.join(' ') }; }
  get textContent() { return this._textOwn + this._children.map(c => c.textContent).join(''); }
  get childNodes() {
    if (this._textOwn) return [{ nodeType: 3, textContent: this._textOwn }, ...this._children];
    return this._children;
  }
  appendChild(c) { c.parentElement = this; this._children.push(c); return c; }
  getAttribute(k) { return this._attrs[k] || null; }
  scrollIntoView() {}
  getBoundingClientRect() {
    if (this._rect) return this._rect;
    // bubble up: row size from parent if container has rect
    return { left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 };
  }
  querySelector(sel) { return (this.querySelectorAll(sel) || [])[0] || null; }
  querySelectorAll(sel) {
    const out = [];
    const sels = sel.split(',').map(s => s.trim());
    const walk = (n) => {
      if (n !== this) {
        for (const s of sels) {
          if (matchOne(n, s)) { out.push(n); break; }
        }
      }
      (n._children || []).forEach(walk);
    };
    walk(this);
    return out;
  }
}
function matchOne(n, sel) {
  // Very limited: 'tag', '.cls', or 'tag.cls'
  let tag = null, cls = null;
  if (sel.startsWith('.')) cls = sel.slice(1);
  else if (sel.includes('.')) [tag, cls] = sel.split('.');
  else tag = sel;
  if (tag) tag = tag.toUpperCase();
  if (tag && n.tagName !== tag) return false;
  if (cls && !n._classes.includes(cls)) return false;
  return true;
}

// Build a synthetic seller page with 3 rows (Steam, Whitebit, Firstmail).
// Each row contains: product-name span, info-icon, cart-icon, mail-icon,
// star-icon, price. Real layout has cart between info and mail.
const root = new El('body');
function makeRow(name, hasCart) {
  const row = new El('div', { cls: 'product-row', rect: { left: 0, top: 100, right: 1000, bottom: 160, width: 1000, height: 60 } });
  row.appendChild(new El('span', { cls: 'product-name', text: name,
    rect: { left: 100, top: 110, right: 400, bottom: 150, width: 300, height: 40 } }));
  row.appendChild(new El('i', { cls: 'fa-info-circle info-icon',
    rect: { left: 410, top: 120, right: 430, bottom: 140, width: 20, height: 20 } }));
  if (hasCart) {
    const btn = new El('button', { cls: 'add-to-cart-btn',
      attrs: { 'aria-label': 'Add to cart' },
      rect: { left: 440, top: 118, right: 470, bottom: 142, width: 30, height: 24 } });
    btn.appendChild(new El('svg', { cls: 'cart-icon shopping-cart',
      rect: { left: 442, top: 120, right: 468, bottom: 140, width: 26, height: 20 } }));
    row.appendChild(btn);
  }
  row.appendChild(new El('i', { cls: 'fa-envelope mail-icon',
    rect: { left: 480, top: 120, right: 500, bottom: 140, width: 20, height: 20 } }));
  row.appendChild(new El('i', { cls: 'fa-star favorite',
    rect: { left: 510, top: 120, right: 530, bottom: 140, width: 20, height: 20 } }));
  row.appendChild(new El('span', { cls: 'price', text: '$0.10' }));
  return row;
}

root.appendChild(makeRow('Steam [Accounts since 2020]', false));
root.appendChild(makeRow('Whitebit [EU]', false));
root.appendChild(makeRow('Firstmail.ltd E-Mail Accounts', true));

global.document = {
  querySelectorAll: (sel) => root.querySelectorAll(sel),
  querySelector:    (sel) => root.querySelector(sel),
};
global.getComputedStyle = () => ({ cursor: 'pointer' });
global.window = { innerHeight: 800, innerWidth: 1920 };

// The actual JS from do_buy_product (product-find part).
const findProduct = ({label}) => {
    const norm = s => (s || '').replace(/\s+/g, ' ').trim();
    const want = norm(label).toLowerCase();
    const all = Array.from(document.querySelectorAll('div, span, p, a, h1, h2, h3, h4, h5, button'));
    const wantPrefix = want.slice(0, Math.min(want.length, 18));
    let matched = all.find(el => {
        const own = norm(Array.from(el.childNodes)
            .filter(n => n.nodeType === 3)
            .map(n => n.textContent).join('')).toLowerCase();
        if (!own) return false;
        return own === want || own.startsWith(wantPrefix) || own.includes(wantPrefix);
    });
    if (!matched) return { ok: false, reason: 'no product text match' };
    const rowRx = /product|card|item|row|tile|listing/i;
    const matchedRect = matched.getBoundingClientRect();
    let row = null;
    let cur = matched.parentElement;
    for (let i = 0; i < 8 && cur; i++) {
        const cls = (cur.className && cur.className.toString) ? cur.className.toString() : '';
        const rr = cur.getBoundingClientRect();
        if (rowRx.test(cls) && rr.width > Math.max(400, matchedRect.width * 1.3)
            && rr.height >= 30 && rr.height < 250) { row = cur; break; }
        cur = cur.parentElement;
    }
    if (!row) {
        cur = matched.parentElement;
        for (let i = 0; i < 8 && cur; i++) {
            const r = cur.getBoundingClientRect();
            if (r.width > 500 && r.height > 40 && r.height < 250
                && r.width > matchedRect.width * 1.3) { row = cur; break; }
            cur = cur.parentElement;
        }
    }
    if (!row) return { ok: false, reason: 'no row ancestor' };
    const cartRx = /(^|[\s_-])cart([\s_-]|$)|shopping[-_]?cart|add[-_]?to[-_]?cart|buy[-_]?now|basket/i;
    const candidates = Array.from(row.querySelectorAll('svg, i, button, a, span, div'));
    const isCarty = (el) => {
        if (!el) return false;
        const cls = (el.className && el.className.toString) ? el.className.toString() : '';
        if (cartRx.test(cls)) return true;
        const al = el.getAttribute && (el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('data-tip') || el.getAttribute('data-tooltip') || '');
        if (al && cartRx.test(al)) return true;
        return false;
    };
    let cart = candidates.find(el => {
        if (!isCarty(el)) return false;
        const r = el.getBoundingClientRect();
        return r.width > 5 && r.height > 5;
    });
    if (cart) {
        let up = cart;
        for (let i = 0; i < 4; i++) {
            if (!up || !up.parentElement) break;
            const p = up.parentElement;
            const tag = p.tagName && p.tagName.toLowerCase();
            if (tag === 'button' || tag === 'a' || (p.onclick != null)) { cart = p; break; }
            up = p;
        }
    }
    if (!cart) return { ok: false, reason: 'no cart icon' };
    const cr = cart.getBoundingClientRect();
    return { ok: true, x: cr.left + cr.width/2, y: cr.top + cr.height/2,
             tag: cart.tagName, cls: cart.className.toString(),
             productText: norm(matched.textContent).slice(0, 80) };
};

console.log('--- Scenario A: full product name ---');
// Debug dump first
const _r3 = root._children[2];
console.log('  Row #2 cls:', _r3.className.toString(), 'rect.w:', _r3.getBoundingClientRect().width);
const _cands = _r3.querySelectorAll('svg, i, button, a, span, div');
console.log('  Candidates in Firstmail row:');
for (const c of _cands) {
  const r = c.getBoundingClientRect();
  console.log('   -', c.tagName, c._classes.join(' '), 'w=' + r.width, 'h=' + r.height);
}
const a = findProduct({label: 'Firstmail.ltd E-Mail Accounts'});
console.log(JSON.stringify(a, null, 2));
console.assert(a.ok === true, 'should find product');
console.assert(/cart/i.test(a.cls), 'should pick the cart-related element, not mail/star');
console.assert(a.x > 430 && a.x < 480, 'click should land on cart icon (x=440-470)');
console.log('PASS');

console.log('\n--- Scenario B: truncated label (mimics what the user sees) ---');
const b = findProduct({label: 'Firstmail.ltd E-Mail Ac...'});
console.log(JSON.stringify(b, null, 2));
console.assert(b.ok === true, 'should find product even with truncated label');
console.log('PASS');

console.log('\n--- Scenario C: nonexistent product ---');
const c = findProduct({label: 'Nonexistent Product XYZ'});
console.log(JSON.stringify(c, null, 2));
console.assert(c.ok === false, 'should report not-found');
console.log('PASS');

console.log('\nAll buy-product scenarios PASSED.');
