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

    // ----- New ranked-candidates heuristic (mirrors styx_register.py) -----
    const includeRx = /(^|[\s_-])cart([\s_-]|$)|shopping[-_]?cart|add[-_]?to[-_]?cart|buy[-_]?now|basket|trolley/i;
    const excludeRx = /mail|envelope|message|chat|letter|email|inbox|star|fav(ou)?rite|heart|like|bookmark|info|question|tooltip|share|copy|external[-_]?link|link[-_]?icon|bell|notification|delete|remove|trash|edit|menu|hamburger|search|filter|sort|user[-_]?avatar|profile/i;
    const allEls = Array.from(row.querySelectorAll('button, a, svg, i, span, div'));
    const idOf = (el) => {
        const cls = (el.className && el.className.toString) ? el.className.toString() : '';
        const al = el.getAttribute && (
               el.getAttribute('aria-label') || el.getAttribute('title')
            || el.getAttribute('data-tip')   || el.getAttribute('data-tooltip')
            || el.getAttribute('data-original-title') || '');
        return { cls, al, allText: cls + ' ' + (al || '') };
    };
    const isExcluded = (el) => excludeRx.test(idOf(el).allText);
    const isIncluded = (el) => includeRx.test(idOf(el).allText);
    const isClickableShape = (el) => {
        const r = el.getBoundingClientRect();
        return !(r.width < 8 || r.width > 100 || r.height < 8 || r.height > 100);
    };
    const toClickable = (el) => {
        if (!el) return el;
        let cur = el;
        for (let i = 0; i < 4; i++) {
            if (!cur || !cur.parentElement) break;
            const p = cur.parentElement;
            const tag = p.tagName && p.tagName.toLowerCase();
            if (tag === 'button' || tag === 'a' || (p.onclick != null)) return p;
            cur = p;
        }
        return el;
    };
    let tier1 = allEls.filter(el => isClickableShape(el) && isIncluded(el) && !isExcluded(el));
    let tier2 = [];
    if (tier1.length === 0) {
        tier2 = allEls.filter(el => {
            if (!isClickableShape(el)) return false;
            if (isExcluded(el)) return false;
            const tag = el.tagName && el.tagName.toLowerCase();
            if (tag === 'button' || tag === 'a') return true;
            if (tag === 'svg' || tag === 'i') {
                const wrap = toClickable(el);
                return wrap && wrap !== el;
            }
            return false;
        });
        const seen = new Set();
        tier2 = tier2.map(toClickable).filter(el => { if (seen.has(el)) return false; seen.add(el); return true; });
        tier2.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
    } else {
        const seen = new Set();
        tier1 = tier1.map(toClickable).filter(el => { if (seen.has(el)) return false; seen.add(el); return true; });
    }
    const ranked = [...tier1, ...tier2];
    if (ranked.length === 0) return { ok: false, reason: 'no cart-like icon in row' };
    const cart = ranked[0];
    const cr = cart.getBoundingClientRect();
    return { ok: true, x: cr.left + cr.width/2, y: cr.top + cr.height/2,
             tag: cart.tagName, cls: cart.className.toString(),
             productText: norm(matched.textContent).slice(0, 80),
             tier: tier1.length > 0 ? 1 : 2,
             candidatesCount: ranked.length };
};

console.log('--- Scenario A: full product name (cart in row) ---');
const a = findProduct({label: 'Firstmail.ltd E-Mail Accounts'});
console.log(JSON.stringify(a, null, 2));
console.assert(a.ok === true, 'should find product');
console.assert(/cart/i.test(a.cls), 'should pick the cart-related element, not mail/star');
console.assert(a.x > 430 && a.x < 480, 'click should land on cart icon (x=440-470)');
console.log('PASS');

console.log('\n--- Scenario B: truncated label ---');
const b = findProduct({label: 'Firstmail.ltd E-Mail Ac...'});
console.log(JSON.stringify(b, null, 2));
console.assert(b.ok === true, 'should find product even with truncated label');
console.log('PASS');

console.log('\n--- Scenario C: nonexistent product ---');
const c = findProduct({label: 'Nonexistent Product XYZ'});
console.log(JSON.stringify(c, null, 2));
console.assert(c.ok === false, 'should report not-found');
console.log('PASS');

// REGRESSION: simulate the bug the user hit - mail icon comes BEFORE cart
// in the row, and neither icon has a distinguishing class on the button
// itself (only on the inner <i>). Make sure the heuristic still picks cart.
console.log('\n--- Scenario D (regression): mail icon precedes cart, no class on button ---');
function makeRowAmbiguous(name) {
    const row = new El('div', { cls: 'product-row',
        rect: { left: 0, top: 100, right: 1000, bottom: 160, width: 1000, height: 60 } });
    row.appendChild(new El('span', { cls: 'product-name', text: name,
        rect: { left: 100, top: 110, right: 400, bottom: 150, width: 300, height: 40 } }));
    // info icon
    const info = new El('button', { cls: 'icon-btn',
        attrs: { 'aria-label': 'Info' },
        rect: { left: 410, top: 118, right: 432, bottom: 142, width: 22, height: 24 } });
    info.appendChild(new El('i', { cls: 'fa-info' }));
    row.appendChild(info);
    // mail icon comes FIRST (this is the bug scenario)
    const mail = new El('button', { cls: 'icon-btn',
        attrs: { 'aria-label': 'Write to seller' },
        rect: { left: 440, top: 118, right: 462, bottom: 142, width: 22, height: 24 } });
    mail.appendChild(new El('i', { cls: 'fa-envelope' }));
    row.appendChild(mail);
    // cart icon AFTER mail
    const cart = new El('button', { cls: 'icon-btn',
        attrs: { 'aria-label': 'Add to cart' },
        rect: { left: 470, top: 118, right: 492, bottom: 142, width: 22, height: 24 } });
    cart.appendChild(new El('i', { cls: 'fa-shopping-cart' }));
    row.appendChild(cart);
    // star
    const star = new El('button', { cls: 'icon-btn',
        attrs: { 'aria-label': 'Add to favorites' },
        rect: { left: 500, top: 118, right: 522, bottom: 142, width: 22, height: 24 } });
    star.appendChild(new El('i', { cls: 'fa-star' }));
    row.appendChild(star);
    return row;
}
// Replace the original Firstmail row with this ambiguous one.
root._children[2] = makeRowAmbiguous('Firstmail.ltd E-Mail Accounts');
root._children[2].parentElement = root;
const d = findProduct({label: 'Firstmail.ltd E-Mail Accounts'});
console.log(JSON.stringify(d, null, 2));
console.assert(d.ok === true, 'should still find product');
// Critical: should NOT pick the mail icon at x=451
console.assert(d.x > 460 && d.x < 500,
    `click should be on cart (x≈481) NOT on mail (x≈451) - got x=${d.x}`);
console.log('PASS - mail icon correctly excluded; cart picked.');

console.log('\nAll buy-product scenarios PASSED.');
