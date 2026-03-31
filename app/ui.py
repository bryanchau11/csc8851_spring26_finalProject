import gradio as gr


def build_demo(*,
               predict,
               update_ingredient_portions,
               gpu_stats_md,
               pipeline_mode,
               food_clf,
               share: bool = True):
    mode_label = {
        'full':   '🚀 Full Pipeline — Phase 3+4+5+6 all active',
        'phase6': '🎯 Phase 6 — Weight-First: YOLO + MiDaS → WeightMLP',
        'phase4': '🔬 Phase 4+5 — YOLO + EfficientNet + Depth + MLP',
        'phase3': '📊 Phase 3 — ResNet-50 baseline',
        None:     '⚠ No model loaded',
    }

    CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --bg:           var(--body-background-fill, #eceffe);
    --surface:      var(--background-fill-primary, #ffffff);
    --surface2:     var(--background-fill-secondary, #f4f5fd);
    --border:       var(--border-color-primary, rgba(160,170,220,0.2));

    --purple:       var(--color-accent, #7b6ef6);
    --purple-light: var(--color-accent-soft, #a89cf7);
    --purple-dark:  var(--color-accent, #5a4de6);
    --purple-dim:   rgba(123,110,246,0.08);
    --purple-glow:  rgba(123,110,246,0.28);

    --green:        var(--color-success, #b5f03c);
    --text:         var(--body-text-color, #1a1d2e);
    --muted:        var(--body-text-color-subdued, #8b90b0);
    --danger:       var(--color-error, #f06292);
    --success:      var(--color-success, #22c4a0);
    --radius:       16px;
    --radius-lg:    24px;
    --shadow:       0 6px 28px rgba(120,130,200,0.13);
    --shadow-hover: 0 12px 44px rgba(120,130,200,0.22);
}

@supports (color: color-mix(in srgb, white 10%, transparent)) {
    :root {
        --purple-dim:  color-mix(in srgb, var(--purple) 12%, transparent);
        --purple-glow: color-mix(in srgb, var(--purple) 35%, transparent);
    }
}

@media (prefers-color-scheme: dark) {
    body, .gradio-container {
        --bg:       var(--body-background-fill, #0b1020);
        --surface:  var(--background-fill-primary, #0f172a);
        --surface2: var(--background-fill-secondary, #111c33);
        --border:   var(--border-color-primary, rgba(255,255,255,0.08));
        --text:     var(--body-text-color, rgba(255,255,255,0.92));
        --muted:    var(--body-text-color-subdued, rgba(255,255,255,0.62));
        --shadow:       0 10px 34px rgba(0,0,0,0.45);
        --shadow-hover: 0 16px 50px rgba(0,0,0,0.55);
    }
}

body.dark, body.dark .gradio-container,
.dark .gradio-container {
    --bg:       var(--body-background-fill, #0b1020);
    --surface:  var(--background-fill-primary, #0f172a);
    --surface2: var(--background-fill-secondary, #111c33);
    --border:   var(--border-color-primary, rgba(255,255,255,0.08));
    --text:     var(--body-text-color, rgba(255,255,255,0.92));
    --muted:    var(--body-text-color-subdued, rgba(255,255,255,0.62));
    --shadow:       0 10px 34px rgba(0,0,0,0.45);
    --shadow-hover: 0 16px 50px rgba(0,0,0,0.55);
}

body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}
footer { display: none !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 28px 32px !important; }

.n5k-hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 32px 40px 28px;
    margin-bottom: 28px;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 24px;
    position: relative;
    overflow: hidden;
}
.n5k-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, var(--purple-glow) 0%, transparent 65%);
    pointer-events: none;
}
.n5k-hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 120px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(181,240,60,0.1) 0%, transparent 65%);
    pointer-events: none;
}

.n5k-hero-left h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    color: var(--text) !important;
    margin: 0 0 6px !important;
    line-height: 1.15 !important;
}
.n5k-hero-left h1 span { color: var(--purple); }
.n5k-hero-left p {
    font-size: 0.88rem !important;
    color: var(--muted) !important;
    margin: 0 0 18px !important;
    font-weight: 400 !important;
}

.n5k-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--purple-dim);
    border: 1px solid rgba(123,110,246,0.2);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--purple);
    margin-right: 6px;
    margin-top: 4px;
}
.n5k-pill.warn {
    background: rgba(240,98,146,0.08);
    border-color: rgba(240,98,146,0.25);
    color: var(--danger);
}
.n5k-pill.ok {
    background: rgba(34,196,160,0.08);
    border-color: rgba(34,196,160,0.25);
    color: var(--success);
}

.n5k-hero-stats {
    display: flex;
    gap: 16px;
    flex-shrink: 0;
}
.n5k-stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 20px;
    text-align: center;
    min-width: 90px;
}
.n5k-stat-chip .val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.04em;
    line-height: 1;
}
.n5k-stat-chip .lbl {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 5px;
}

.n5k-section-label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    margin: 0 0 10px !important;
}

#img_upload {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 2px dashed rgba(123,110,246,0.25) !important;
    background: var(--surface) !important;
    min-height: 320px !important;
    box-shadow: var(--shadow) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
#img_upload:hover {
    border-color: rgba(123,110,246,0.55) !important;
    box-shadow: var(--shadow-hover) !important;
}
#img_upload .icon-wrap svg { color: var(--purple) !important; }
#img_upload label span { color: var(--muted) !important; font-family: 'Manrope', sans-serif !important; }

#predict_btn {
    background: linear-gradient(135deg, var(--purple-dark), var(--purple-light)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    padding: 16px 0 !important;
    width: 100% !important;
    cursor: pointer !important;
    box-shadow: 0 6px 24px var(--purple-glow) !important;
    transition: box-shadow 0.25s, transform 0.15s, filter 0.2s !important;
}
#predict_btn:hover {
    filter: brightness(1.08) !important;
    box-shadow: 0 10px 36px var(--purple-glow) !important;
    transform: translateY(-2px) !important;
}
#predict_btn:active { transform: translateY(0) !important; }

#text_out {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px 22px !important;
    font-size: 0.9rem !important;
    line-height: 1.75 !important;
    min-height: 110px !important;
    color: var(--text) !important;
    box-shadow: var(--shadow) !important;
}
#text_out p, #text_out strong { font-family: 'Manrope', sans-serif !important; color: var(--text) !important; }
#text_out strong { color: var(--purple) !important; }

#json_out {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-size: 0.8rem !important;
    color: var(--purple-dark) !important;
    box-shadow: inset 0 2px 8px rgba(120,130,200,0.07) !important;
}
#json_out .json-holder { background: transparent !important; }

.n5k-html-table th { border: none !important; }
.n5k-html-table td {
    border: none !important;
    border-bottom: 1px solid var(--border) !important;
}
.n5k-html-table tr:last-child td { border-bottom: none !important; }
.n5k-row { transition: background 0.15s !important; }
.n5k-row:hover { background: var(--purple-dim) !important; }

.n5k-accordion > .label-wrap {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 14px 18px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    cursor: pointer !important;
    box-shadow: var(--shadow) !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
.n5k-accordion > .label-wrap:hover {
    background: var(--purple-dim) !important;
    box-shadow: var(--shadow-hover) !important;
}
.n5k-accordion > .label-wrap svg { color: var(--purple) !important; }
.n5k-accordion .inner {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius) var(--radius) !important;
    padding: 18px !important;
}
.n5k-accordion .inner p,
.n5k-accordion .inner li,
.n5k-accordion .inner td { color: var(--text) !important; font-family: 'Manrope', sans-serif !important; font-size: 0.86rem !important; }
.n5k-accordion .inner th { color: var(--muted) !important; font-family: 'Outfit', sans-serif !important; font-size: 0.7rem !important; }

.n5k-tips {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--purple) !important;
    border-radius: var(--radius) !important;
    padding: 16px 22px !important;
    font-size: 0.83rem !important;
    color: var(--muted) !important;
    line-height: 1.75 !important;
    box-shadow: var(--shadow) !important;
}
.n5k-tips p { color: var(--muted) !important; margin: 0 !important; }
.n5k-tips strong { color: var(--purple) !important; }
.n5k-tips code {
    background: var(--purple-dim);
    color: var(--purple-dark);
    border-radius: 5px;
    padding: 1px 6px;
    font-size: 0.8rem;
}

.n5k-table-header {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: var(--radius) var(--radius) 0 0;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.n5k-table-header .t-icon {
    width: 34px; height: 34px;
    background: var(--purple-dim);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.n5k-table-header .t-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
.n5k-table-header .t-sub {
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 2px;
}

.gradio-container .prose { color: var(--text) !important; }
.gradio-container label > span {
    color: var(--muted) !important;
    font-family: 'Manrope', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
}
.gradio-container .gap { gap: 20px !important; }
"""

    _active_mode = mode_label.get(pipeline_mode, '⚠ No model loaded')
    _clf_ok = food_clf is not None
    _p6_banner = ''
    if pipeline_mode == 'phase6':
        _p6_banner = (
            '<div style="margin-top:14px;padding:10px 16px;background:rgba(123,110,246,0.07);'
            'border:1px solid rgba(123,110,246,0.18);border-radius:10px;font-size:0.78rem;color:var(--purple);">'
            '⚡ <strong>Phase 6 active</strong>: Dish weight predicted first → converted to calories/macros '
            'via calorie-density constants derived from Nutrition5K (Professor spec §4.3).</div>'
        )

    HERO_HTML = f"""
<div class="n5k-hero">
  <div class="n5k-hero-left">
    <h1>🥗 <span>Nutrition</span>5K</h1>
    <p>Upload any real-world food photo — get calories, macros &amp; ingredient breakdown instantly.</p>
    <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
      <span class="n5k-pill">🚀 {pipeline_mode or 'no model'}</span>
      <span class="n5k-pill {'ok' if _clf_ok else 'warn'}">
        {'🍕 Classifier active' if _clf_ok else '⚠ No classifier'}
      </span>
      <span class="n5k-pill" style="font-size:0.7rem;opacity:0.75;max-width:380px;
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
        {_active_mode}
      </span>
    </div>
    {_p6_banner}
  </div>
  <div class="n5k-hero-stats">
    <div class="n5k-stat-chip">
      <div class="val">101</div>
      <div class="lbl">Food Classes</div>
    </div>
    <div class="n5k-stat-chip">
      <div class="val">6</div>
      <div class="lbl">Phases</div>
    </div>
    <div class="n5k-stat-chip">
      <div class="val" style="color:var(--purple);">{'✓' if _clf_ok else '✗'}</div>
      <div class="lbl">Classifier</div>
    </div>
  </div>
</div>
"""

    with gr.Blocks(title='Nutrition5K Estimator', css=CUSTOM_CSS) as demo:
        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                gr.HTML('<p class="n5k-section-label">📸 Food Image</p>')
                img_input = gr.Image(type='pil', label='', elem_id='img_upload')
                predict_btn = gr.Button('⚡  Analyse Dish', variant='primary', elem_id='predict_btn')

            with gr.Column(scale=7):
                gr.HTML('<p class="n5k-section-label">🧪 Nutrition Prediction</p>')
                text_out = gr.HTML(
                    value='<div style="color: var(--muted); padding: 20px;"><em>Upload a dish photo and press <strong>Analyse Dish</strong> to see results.</em></div>',
                    elem_id='text_out',
                )

                with gr.Accordion('📦 Raw JSON Values', open=False, elem_classes='n5k-accordion'):
                    json_out = gr.JSON(label='', elem_id='json_out')

        gr.HTML("""
    <div style="margin-top:20px;">
      <div class="n5k-table-header">
        <div class="t-icon">🍽</div>
        <div>
          <div class="t-title">Detected Dish &amp; Calorie Breakdown</div>
          <div class="t-sub">Phase 5 dish classifier (Food-101) + YOLO segmentation</div>
        </div>
      </div>
    </div>
    """)
        ingredient_table = gr.HTML()

        gr.HTML('<p class="n5k-section-label" style="margin-top:14px;">🥄 Edit Ingredient Portions</p>')

        _EDIT_UNITS = ['g', 'oz', 'cup', 'tbsp', 'tsp']
        _MAX_EDIT_ROWS = 10

        portion_name_inputs = []
        portion_amt_inputs = []
        portion_unit_inputs = []

        for _i in range(_MAX_EDIT_ROWS):
            with gr.Row():
                nm = gr.Textbox(label='Ingredient', value='', interactive=False, visible=False)
                amt = gr.Number(label='Amount', value=0, precision=0, interactive=True, visible=False)
                un = gr.Dropdown(label='Unit', choices=_EDIT_UNITS, value='g', interactive=True, visible=False)
            portion_name_inputs.append(nm)
            portion_amt_inputs.append(amt)
            portion_unit_inputs.append(un)

        update_ing_btn = gr.Button('Update Ingredients')
        gr.HTML(
            '<div style="font-size:0.78rem; color:var(--muted); margin:6px 0 10px;">'
            'Units are converted to grams (cups/tbsp/tsp are rough).</div>'
        )

        def _predict_and_fill_rows(image):
            label_html, result_json, table_html, weight_md, gpu_md, defaults = predict(image)

            if defaults is None:
                defaults = []
            try:
                defaults = list(defaults)
            except Exception:
                defaults = []

            updates = []
            for i in range(_MAX_EDIT_ROWS):
                if i < len(defaults) and isinstance(defaults[i], (list, tuple)) and len(defaults[i]) >= 3:
                    dish, amt, unit = defaults[i][0], defaults[i][1], defaults[i][2]
                    unit = unit if str(unit) in _EDIT_UNITS else 'g'
                    updates.append(gr.update(value=str(dish), visible=True))
                    updates.append(gr.update(value=amt, visible=True))
                    updates.append(gr.update(value=str(unit), visible=True))
                else:
                    updates.append(gr.update(value='', visible=False))
                    updates.append(gr.update(value=0, visible=False))
                    updates.append(gr.update(value='g', visible=False))

            return label_html, result_json, table_html, weight_md, gpu_md, *updates

        def _update_from_row_controls(result_json, *vals):
            rows = []
            # vals order: name0..nameN, amt0..amtN, unit0..unitN (we pass grouped below)
            # but to keep wiring simple, we pass (name, amt, unit) per row sequentially.
            for i in range(_MAX_EDIT_ROWS):
                name = vals[i * 3 + 0] if (i * 3 + 0) < len(vals) else ''
                amt = vals[i * 3 + 1] if (i * 3 + 1) < len(vals) else 0
                unit = vals[i * 3 + 2] if (i * 3 + 2) < len(vals) else 'g'
                if not str(name or '').strip():
                    continue
                rows.append([str(name), amt, str(unit or 'g')])
            return update_ingredient_portions(result_json, rows)

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion('⚖  Weight Prediction Details', open=False, elem_classes='n5k-accordion'):
                    weight_detail_out = gr.Markdown(value='_Run a prediction to see the weight breakdown._')
            with gr.Column(scale=1):
                with gr.Accordion('⚡  GPU / Compute Stats', open=False, elem_classes='n5k-accordion'):
                    gpu_stats_out = gr.Markdown(value=gpu_stats_md())

        gr.HTML("""
    <div class="n5k-tips" style="margin-top:16px;">
      <p>
        <strong>Tips</strong> — Works best with a single dish centred in frame.
        Food-101 covers pizza, sushi, ramen, burgers, salads, pasta, tacos, and 94 more categories.
        For Asian &amp; regional foods, train on UECFOOD-256 or iFood-2019 (see notebook 05).
        <br><br>
        <strong>Phase 6 setup</strong> — Train <code>06_weight_prediction.ipynb</code> on Kaggle,
        then download <code>best_weight_mlp.pt</code>, <code>nutrition_constants.json</code>,
        and <code>weight_feat_stats.npz</code> into <code>models/</code>.
      </p>
    </div>
    """)

        _portion_row_controls = []
        for i in range(_MAX_EDIT_ROWS):
            _portion_row_controls.extend([portion_name_inputs[i], portion_amt_inputs[i], portion_unit_inputs[i]])

        predict_btn.click(
            fn=_predict_and_fill_rows,
            inputs=img_input,
            outputs=[text_out, json_out, ingredient_table, weight_detail_out, gpu_stats_out, *_portion_row_controls],
        )

        update_ing_btn.click(
            fn=_update_from_row_controls,
            inputs=[json_out, *_portion_row_controls],
            outputs=[text_out, json_out, ingredient_table],
        )

    return demo
