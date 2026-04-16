/**
 * Computes a viewport-safe { top, left } for a position:fixed tooltip.
 *
 * @param {DOMRect|{left,top,right,bottom,width,height}} triggerRect
 *   Bounding rect of the trigger element (from getBoundingClientRect).
 *   For cursor-relative tooltips, pass a zero-size rect at the cursor:
 *     { left: cx, top: cy, right: cx, bottom: cy, width: 0, height: 0 }
 *
 * @param {object} opts
 *   prefer  – 'above' | 'below' | 'right'  (initial placement, flipped if it would overflow)
 *   gap     – px gap between trigger and tooltip edge  (default 8)
 *   width   – estimated tooltip width in px            (default 240)
 *   height  – estimated tooltip height in px           (default 36)
 *
 * @returns {{ top: number, left: number }}  — use directly in style={{ position:'fixed', ...pos }}
 */
const PAD = 8; // minimum distance from each viewport edge

export function tooltipPosition(triggerRect, { prefer = 'above', gap = 8, width = 240, height = 36 } = {}) {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const r  = triggerRect;

  let top, left;

  if (prefer === 'right') {
    // Tooltip to the right of the trigger, vertically centred on it
    left = r.right + gap;
    top  = r.top + r.height / 2 - height / 2;
    // Flip left if it would overflow the right edge
    if (left + width > vw - PAD) left = r.left - gap - width;
    // Clamp vertically
    top = Math.max(PAD, Math.min(vh - PAD - height, top));
  } else {
    // Centre horizontally over the trigger
    left = r.left + r.width / 2 - width / 2;

    if (prefer === 'above') {
      top = r.top - gap - height;
      if (top < PAD) top = r.bottom + gap;          // flip below
    } else {                                          // 'below'
      top = r.bottom + gap;
      if (top + height > vh - PAD) top = r.top - gap - height; // flip above
    }
  }

  // Always clamp horizontal so tooltip never exits viewport
  left = Math.max(PAD, Math.min(vw - PAD - width, left));

  return { top: Math.round(top), left: Math.round(left) };
}
