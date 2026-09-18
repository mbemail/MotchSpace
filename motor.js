// motor.js - the shared input deck and save RAM for MotchSpace carts.
//
// One copy for every cart, so a fix here lands everywhere. A cart builds a controller with
// MOTCH.create() and reads the pad the same way it always did:
//
//     const ctl = MOTCH.create({ slot:'galactica', keys:KEYS, buttons:{...}, stick:true,
//                                unlock:audioUnlock, lift:34 })
//     ctl.pad.L, ctl.pad.A ...        held state, exactly as before
//     ctl.hit('START')                fresh press this frame
//     ctl.sync()                      at the end of every tick, to latch the edges
//
// Touch is a held-drag stick, not a flick: dragging sets pad.ptr and pad.px/pad.py in canvas
// pixels, so a cart can steer toward the finger instead of guessing a direction from a swipe.
// Anything that is not the stick still goes through down()/up(), so every input path - key,
// button, drag - ends in the same place.
window.MOTCH = (() => {
  'use strict'

  const store = {                                 // localStorage throws in private mode
    get: k => { try { return localStorage.getItem(k) } catch { return null } },
    set: (k, v) => { try { localStorage.setItem(k, v) } catch {} }
  }

  const CSS = `
    html,body{touch-action:none;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
    #pad{position:fixed;inset:auto 0 0 0;display:none;justify-content:space-between;align-items:flex-end;
      padding:.6rem calc(.6rem + env(safe-area-inset-right)) calc(.6rem + env(safe-area-inset-bottom))
              calc(.6rem + env(safe-area-inset-left));z-index:9;pointer-events:none}
    body.touch #pad{display:flex}
    #pad .grp{display:flex;gap:.5rem;pointer-events:auto}
    #pad button{font:700 13px/1 monospace;letter-spacing:.12em;color:#BCBCBC;background:#000;
      border:2px solid #7C7C7C;padding:.9rem 1.2rem;border-radius:0;touch-action:none}
    #pad button:active{background:#B8F818;border-color:#B8F818;color:#000}
    @media (max-width:360px){#pad button{padding:.7rem .8rem;font-size:11px}}`

  function create(cfg){
    const cv   = document.querySelector(cfg.canvas || '#tv') || document.querySelector('canvas')
    const pad  = {}, was = {}, OPP = {L:'R', R:'L', U:'D', D:'U'}
    const opt  = { input: cfg.coarse ? 'TOUCH' : 'KEYS', sound: 'ON' }
    try { Object.assign(opt, JSON.parse(store.get(cfg.slot + 'Opt') || '{}')) } catch {}
    const save = () => store.set(cfg.slot + 'Opt', JSON.stringify(opt))

    // ---- the one place every input lands -------------------------------------
    function down(b){
      if (!b) return
      if (cfg.unlock) cfg.unlock()                 // creates the AudioContext, or resumes it
      pad[b] = 1
      if (OPP[b]) pad[OPP[b]] = 0                  // a d-pad cannot press opposites
    }
    const up = b => { pad[b] = 0 }
    const hit = b => pad[b] && !was[b]
    const sync = () => Object.assign(was, pad)

    // ---- keyboard ------------------------------------------------------------
    const KEYS = cfg.keys || {}
    addEventListener('keydown', e => {
      const b = KEYS[e.code]
      if (!b) return
      e.preventDefault()
      down(b)
    })
    addEventListener('keyup', e => { const b = KEYS[e.code]; if (b) up(b) })
    addEventListener('blur', () => { for (const k in pad) pad[k] = 0; pad.ptr = 0 })

    // ---- on-screen deck ------------------------------------------------------
    const style = document.createElement('style')
    style.textContent = CSS
    document.head.append(style)

    const deck = document.createElement('div')
    deck.id = 'pad'
    for (const side of ['left', 'right']) {
      const grp = document.createElement('div')
      grp.className = 'grp'
      for (const btn of (cfg.buttons && cfg.buttons[side]) || []) {
        const el = document.createElement('button')
        el.type = 'button'
        el.dataset.b = btn.b
        el.textContent = btn.label
        el.addEventListener('pointerdown', e => {
          e.preventDefault()
          if (el.setPointerCapture) el.setPointerCapture(e.pointerId)
          down(btn.b)
        })
        const release = e => { e.preventDefault(); up(btn.b) }
        el.addEventListener('pointerup', release)
        el.addEventListener('pointercancel', release)
        el.addEventListener('contextmenu', e => e.preventDefault())
        grp.append(el)
      }
      deck.append(grp)
    }
    document.body.append(deck)

    // ---- the page is a held-drag stick --------------------------------------
    // Lift moves the ship above the fingertip so a thumb never covers it. Only one pointer
    // drives the stick; a second finger on FIRE must not yank the ship sideways.
    let stickId = null
    const toCanvas = e => {
      const r = cv.getBoundingClientRect()
      return { x: (e.clientX - r.left) / r.width * 256,
               y: (e.clientY - r.top) / r.height * 240 - (cfg.lift || 0) }
    }
    const onStick = e => { const p = toCanvas(e); pad.ptr = 1; pad.px = p.x; pad.py = p.y }
    const bad = e => opt.input !== 'TOUCH' || (e.target && e.target.closest && e.target.closest('#pad'))

    cv.addEventListener('pointerdown', e => {
      if (bad(e) || stickId !== null) return
      e.preventDefault()
      stickId = e.pointerId
      if (cv.setPointerCapture) cv.setPointerCapture(e.pointerId)
      onStick(e)
    })
    cv.addEventListener('pointermove', e => { if (e.pointerId === stickId) { e.preventDefault(); onStick(e) } })
    for (const ev of ['pointerup', 'pointercancel'])   // capture keeps the drag alive off-canvas
      cv.addEventListener(ev, e => {
        if (e.pointerId !== stickId) return
        stickId = null
        pad.ptr = 0
      })

    // ---- INPUT option ---------------------------------------------------------
    const apply = () => document.body.classList.toggle('touch', opt.input === 'TOUCH')
    const setInput = v => { opt.input = v; save(); apply() }
    apply()

    return { pad, opt, save, hit, sync, down, up, setInput, canvas: cv }
  }

  const COARSE = typeof matchMedia === 'function' && matchMedia('(pointer:coarse)').matches
  return { create, COARSE, store }
})()
