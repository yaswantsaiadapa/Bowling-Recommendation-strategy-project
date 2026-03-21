"""
app.py — Cricket Strategy Recommendation System
Flask web dashboard

Routes:
  GET  /                        → Home / search
  GET  /batsman/<name>          → Batsman pre-game report
  GET  /bowler/<name>           → Bowler pre-game report
  GET  /matchup                 → Head-to-head matchup report  (?bat=X&bowl_style=Y&phase=Z)
  GET  /insights                → Data-wide insights dashboard
  GET  /api/players             → JSON list of all player names (for autocomplete)
  GET  /api/batsman/<name>      → JSON batsman summary
  GET  /api/bowler/<name>       → JSON bowler summary
"""

import os
import json
import sys

# Make sure analytics package is importable from app.py location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, abort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics.data_loader import (
    get_df, get_phase_sr, get_profiles, get_bowler_stats, get_matchup,
    all_batsman_names, all_bowler_names
)
import analytics.batsman  as bat_analytics
import analytics.bowler   as bowl_analytics
import analytics.strategy as strat

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _decode_bdata(obj):
    """Recursively convert Plotly typed-array dicts {dtype, bdata} to plain Python lists.
    Newer Plotly (>5.12) encodes numeric arrays as base64 binary for performance.
    Plotly.js handles it, but the conversion ensures compatibility across all versions.
    """
    import base64, struct
    if isinstance(obj, dict):
        if 'bdata' in obj and 'dtype' in obj:
            dtype_map = {'f8': ('d',8), 'f4': ('f',4), 'i4': ('i',4),
                         'i8': ('q',8), 'u1': ('B',1), 'u4': ('I',4)}
            raw          = base64.b64decode(obj['bdata'])
            fmt_ch, size = dtype_map.get(obj['dtype'], ('d', 8))
            count        = len(raw) // size
            return list(struct.unpack_from(f'{count}{fmt_ch}', raw))
        return {k: _decode_bdata(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_decode_bdata(i) for i in obj]
    return obj


def _fig_to_json(fig):
    """Convert a Plotly figure to a plain-Python dict for Jinja tojson.
    - Applies dark theme
    - Decodes binary-encoded numeric arrays to plain lists so Plotly.js
      renders data points correctly in all browser environments.
    """
    fig.update_layout(**_DARK)
    raw = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    return _decode_bdata(raw)

import plotly

# Dark theme applied to every chart
_DARK = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,29,46,0.9)',
    font=dict(color='#e8eaf6'),
    xaxis=dict(gridcolor='#2e3250', tickfont=dict(color='#8891b2')),
    yaxis=dict(gridcolor='#2e3250', tickfont=dict(color='#8891b2')),
)


def zone_heatmap_json(zone_df, score_col, title, colorscale):
    if zone_df is None or zone_df.empty:
        return None
    pivot = zone_df.pivot_table(score_col, index='pitchLength', columns='pitchLine').fillna(0)
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.tolist(),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        text=[[round(v, 2) for v in row] for row in pivot.values.tolist()],
        texttemplate='%{text}',
        showscale=True
    ))
    fig.update_layout(title=title, height=320, margin=dict(l=10,r=10,t=40,b=10),
                      xaxis_title='Pitch Line', yaxis_title='Pitch Length', **_DARK)
    return _fig_to_json(fig)


def bar_json(x, y, title, color=None, orientation='v', colorscale=None):
    if color and colorscale:
        fig = px.bar(x=x, y=y, color=color, color_continuous_scale=colorscale,
                     title=title, orientation=orientation)
    else:
        fig = px.bar(x=x, y=y, title=title, orientation=orientation)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=340,
                      showlegend=False)
    return _fig_to_json(fig)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Home page with search bars."""
    batsmen = all_batsman_names()
    bowlers = all_bowler_names()
    return render_template('index.html', batsmen=batsmen, bowlers=bowlers)


@app.route('/batsman/<path:name>')
def batsman_report(name):
    """Full batsman pre-game report page."""
    name = name.strip()

    stats = bat_analytics.overall_stats(name)
    if stats is None:
        abort(404, description=f"No data found for batsman '{name}'")

    sz   = bat_analytics.strength_zones(name)
    wz   = bat_analytics.weakness_zones(name)
    srp  = bat_analytics.shot_risk_profile(name)
    vbs  = bat_analytics.vs_bowling_style(name)
    cvs  = bat_analytics.chase_vs_set(name)
    pp   = bat_analytics.pressure_performance(name)
    gp   = bat_analytics.ground_performance(name)
    h2h  = bat_analytics.head_to_head_vs_bowlers(name)
    ps   = bat_analytics.phase_stats(name)

    # ── Build charts ──
    charts = {}

    # Strength heatmap
    charts['strength_heat'] = zone_heatmap_json(sz, 'strength_score',
        'Strength Zone Map (High = Scores Freely)', 'Greens')

    # Weakness heatmap
    charts['weakness_heat'] = zone_heatmap_json(wz, 'weakness_score',
        'Weakness Zone Map (High = Vulnerable)', 'Reds')

    # Shot risk scatter
    if not srp.empty:
        fig = px.scatter(srp, x='wicket_pct', y='boundary_pct',
                         size='balls', text='shot_label', color='risk_reward',
                         color_continuous_scale='RdYlGn',
                         title='Shot Risk Profile (right = risky, top = rewarding)',
                         labels={'wicket_pct': 'Wicket %', 'boundary_pct': 'Boundary %'})
        fig.update_traces(textposition='top center')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=360)
        charts['shot_risk'] = _fig_to_json(fig)

    # vs Bowling Style
    if not vbs.empty:
        vbs_s = vbs.sort_values('strike_rate')
        fig = px.bar(vbs_s, y='Bowler_Bowling_Style', x='strike_rate',
                     orientation='h', color='wicket_pct',
                     color_continuous_scale='RdYlGn_r',
                     title='SR vs Each Bowling Style (colour = wicket%)',
                     text=vbs_s['strike_rate'])
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=80), height=400,
                          yaxis_title='', coloraxis_showscale=False)
        charts['vs_style'] = _fig_to_json(fig)

    # Phase breakdown
    if not ps.empty:
        ps_c = ps.copy()
        ps_c['match_phase'] = pd.Categorical(
            ps_c['match_phase'], ['Powerplay','Middle','Death'], ordered=True)
        ps_c = ps_c.sort_values('match_phase')
        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=['Strike Rate','Boundary %','Dot Ball %'])
        colors = {'Powerplay':'#2ecc71','Middle':'#e67e22','Death':'#e74c3c'}
        for i, col in enumerate(['strike_rate','boundary_pct','dot_ball_pct'], 1):
            fig.add_trace(go.Bar(
                x=ps_c['match_phase'], y=ps_c[col],
                marker_color=[colors[p] for p in ps_c['match_phase']],
                text=ps_c[col].round(1), textposition='outside',
                showlegend=False), row=1, col=i)
        fig.update_layout(title='Phase Breakdown', height=360,
                           margin=dict(l=10,r=10,t=40,b=10))
        charts['phase'] = _fig_to_json(fig)

    # Chase vs Set
    if not cvs.empty:
        fig = px.bar(cvs, x='label', y=['strike_rate','boundary_pct'],
                     barmode='group', title='Chase vs Set Performance',
                     labels={'value':'Rate','variable':'Metric','label':''})
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320)
        charts['chase_vs_set'] = _fig_to_json(fig)

    # Pressure performance
    if not pp.empty:
        fig = px.bar(pp, x='pressure_band', y='strike_rate',
                     color='strike_rate', color_continuous_scale='RdYlGn',
                     title='SR Under Wicket Pressure',
                     text=pp['strike_rate'])
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320,
                           coloraxis_showscale=False)
        charts['pressure'] = _fig_to_json(fig)

    # Ground performance
    if not gp.empty:
        fig = px.bar(gp.head(10), x='Ground Name', y='strike_rate',
                     color='boundary_pct', color_continuous_scale='Blues',
                     title='Ground Performance (Top 10 by SR)',
                     text=gp.head(10)['strike_rate'])
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=80),
                           height=360, xaxis_tickangle=-35)
        charts['ground'] = _fig_to_json(fig)

    # Head-to-head
    if not h2h.empty:
        fig = px.bar(h2h.head(10), x='Bowler_Name', y='success_index_scaled',
                     color='Bowler_Bowling_Style',
                     hover_data=['economy_rate','wicket_pct'],
                     title='Most Effective Bowlers vs This Batsman')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=60),
                           height=360, xaxis_tickangle=-30)
        charts['h2h'] = _fig_to_json(fig)

    return render_template('batsman_report.html',
                            name=stats['full_name'],
                            stats=stats,
                            charts=charts,
                            sz=sz.head(3).to_dict('records') if not sz.empty else [],
                            wz=wz.head(3).to_dict('records') if not wz.empty else [])


@app.route('/bowler/<path:name>')
def bowler_report(name):
    """Full bowler pre-game report page."""
    name = name.strip()

    stats = bowl_analytics.overall_stats(name)
    if stats is None:
        abort(404, description=f"No data found for bowler '{name}'")

    sz   = bowl_analytics.strength_zones(name)
    wz   = bowl_analytics.weakness_zones(name)
    ps   = bowl_analytics.phase_stats(name)
    vbs  = bowl_analytics.vs_batting_style(name)
    et   = bowl_analytics.economy_trend(name)
    wdp  = bowl_analytics.wicket_delivery_profile(name)
    tgt  = bowl_analytics.best_batsmen_to_target(name)

    charts = {}

    charts['strength_heat'] = zone_heatmap_json(sz, 'success_score',
        'Strength Zone Map (High = Economy + Wickets)', 'Greens')

    charts['weakness_heat'] = zone_heatmap_json(wz, 'weakness_score',
        'Weakness Zone Map (High = Leaks Runs)', 'Reds')

    if not ps.empty:
        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=['Economy','Wicket %','Dot Ball %'])
        colors = {'Powerplay':'#2ecc71','Middle':'#e67e22','Death':'#e74c3c'}
        for i, col in enumerate(['economy','wicket_pct','dot_pct'], 1):
            fig.add_trace(go.Bar(
                x=ps['match_phase'], y=ps[col],
                marker_color=[colors.get(p,'#999') for p in ps['match_phase']],
                text=ps[col].round(2), textposition='outside',
                showlegend=False), row=1, col=i)
        fig.update_layout(title='Phase Performance', height=360,
                           margin=dict(l=10,r=10,t=40,b=10))
        charts['phase'] = _fig_to_json(fig)

    if not et.empty:
        fig = px.line(et, x='over_int', y='economy', markers=True,
                      title='Economy Trend Across Overs',
                      labels={'over_int':'Over','economy':'Economy Rate'})
        fig.add_hline(y=et['economy'].mean(), line_dash='dot',
                      line_color='gray', annotation_text='Average')
        fig.add_vrect(x0=16, x1=20, fillcolor='lightsalmon', opacity=0.2,
                      layer='below', line_width=0, annotation_text='Death')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320)
        charts['econ_trend'] = _fig_to_json(fig)

    if not vbs.empty:
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Economy','Wicket %'])
        clrs = ['#3498db','#e74c3c']
        for i, col in enumerate(['economy','wicket_pct'], 1):
            fig.add_trace(go.Bar(x=vbs['Batsman_Batting_Style'], y=vbs[col],
                                  marker_color=clrs, text=vbs[col],
                                  textposition='outside', showlegend=False), row=1, col=i)
        fig.update_layout(title='vs LHB vs RHB', height=340,
                           margin=dict(l=10,r=10,t=40,b=10))
        charts['vs_style'] = _fig_to_json(fig)

    if not wdp.empty:
        top = wdp.head(10).copy()
        top['zone'] = top['pitchLine'].str[:8] + ' / ' + top['pitchLength'].str[:5] + ' / ' + top['match_phase']
        fig = px.bar(top, x='zone', y='wicket_pct',
                     color='wickets', color_continuous_scale='Reds',
                     title='Top Wicket-Taking Zones',
                     text=top['wickets'], labels={'zone':'Zone','wicket_pct':'Wicket %'})
        fig.update_traces(textposition='outside')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=80),
                           height=360, xaxis_tickangle=-30)
        charts['wicket_zones'] = _fig_to_json(fig)

    if not tgt.empty:
        fig = px.bar(tgt.head(8), x='Batsman_Name', y='success_index_scaled',
                     color='performance_label', title='Best Batsmen to Target',
                     hover_data=['economy_rate','wicket_pct'])
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=60),
                           height=340, xaxis_tickangle=-25)
        charts['targets'] = _fig_to_json(fig)

    return render_template('bowler_report.html',
                            name=stats['full_name'],
                            stats=stats,
                            charts=charts,
                            sz=sz.head(3).to_dict('records') if not sz.empty else [],
                            wz=wz.head(3).to_dict('records') if not wz.empty else [])


@app.route('/matchup')
def matchup():
    """Matchup report: ?bat=Virat Kohli&bowl_style=right-arm fast&phase=Middle"""
    bat_name   = request.args.get('bat', '').strip()
    bowl_style = request.args.get('bowl_style', 'right-arm fast').strip()
    phase      = request.args.get('phase', 'Middle').strip()

    batsmen = all_batsman_names()

    if not bat_name:
        return render_template('matchup.html', batsmen=batsmen,
                                report=None, bat_name='', bowl_style=bowl_style, phase=phase)

    report = strat.full_matchup_report(bat_name, phase, bowl_style)

    # Best bowlers vs this batsman
    best_bowlers = strat.find_best_bowler_vs(bat_name, top_n=8)
    bb_chart = None
    if not best_bowlers.empty:
        fig = px.bar(best_bowlers, x='Bowler_Name', y='success_index_scaled',
                     color='Bowler_Bowling_Style',
                     hover_data=['economy_rate','wicket_pct'],
                     title=f'Most Effective Bowlers vs {bat_name}')
        fig.update_layout(margin=dict(l=10,r=10,t=40,b=60),
                           height=340, xaxis_tickangle=-25)
        bb_chart = _fig_to_json(fig)

    return render_template('matchup.html',
                            batsmen=batsmen,
                            report=report,
                            bat_name=bat_name,
                            bowl_style=bowl_style,
                            phase=phase,
                            bb_chart=bb_chart)


@app.route('/insights')
def insights():
    """Global insights dashboard."""
    df = get_df()
    charts = {}

    # Delivery outcomes pie
    outcomes = {
        'Dot Ball': int((df['run'] == 0).sum()),
        'Single/Double': int(((df['run'] >= 1) & (df['isBoundary'] == 0) & (df['isWicket'] == 0)).sum()),
        'Boundary': int(df['isBoundary'].sum()),
        'Wicket': int(df['isWicket'].sum())
    }
    fig = px.pie(names=list(outcomes.keys()), values=list(outcomes.values()),
                  title='Overall Delivery Outcome Distribution',
                  color_discrete_sequence=px.colors.qualitative.Safe, hole=0.35)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320)
    charts['outcomes'] = _fig_to_json(fig)

    # Phase avg runs
    phase_stats = df.groupby('match_phase').agg(
        avg_run=('run','mean'), bdry=('isBoundary','mean'),
        wkt=('isWicket','mean'), dot=('run', lambda x:(x==0).mean())
    ).reindex(['Powerplay','Middle','Death']).reset_index()
    phase_stats[['bdry','wkt','dot']] *= 100

    fig = make_subplots(rows=1, cols=4,
        subplot_titles=['Avg Runs/Ball','Boundary %','Wicket %','Dot Ball %'])
    colors = ['#2ecc71','#e67e22','#e74c3c']
    for i, col in enumerate(['avg_run','bdry','wkt','dot'], 1):
        fig.add_trace(go.Bar(x=phase_stats['match_phase'], y=phase_stats[col].round(2),
                              marker_color=colors, showlegend=False,
                              text=phase_stats[col].round(2), textposition='outside'),
                      row=1, col=i)
    fig.update_layout(title='Phase Comparison', height=380,
                       margin=dict(l=10,r=10,t=40,b=10))
    charts['phase_compare'] = _fig_to_json(fig)

    # Pitch zone heatmap avg runs
    pivot_run = df.pivot_table('run', index='pitchLength', columns='pitchLine', aggfunc='mean')
    fig = px.imshow(pivot_run.round(2), text_auto='.2f', color_continuous_scale='YlOrRd',
                     title='Avg Runs — Pitch Length × Line', aspect='auto')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320)
    charts['zone_run'] = _fig_to_json(fig)

    # Pitch zone heatmap wicket%
    pivot_wkt = df.pivot_table('isWicket', index='pitchLength', columns='pitchLine',
                                 aggfunc='mean') * 100
    fig = px.imshow(pivot_wkt.round(2), text_auto='.2f', color_continuous_scale='Reds',
                     title='Wicket % — Pitch Length × Line', aspect='auto')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=320)
    charts['zone_wkt'] = _fig_to_json(fig)

    # Top bowler styles
    bs = get_bowler_stats()
    bs_plot = bs[bs['balls_bowled'] >= 50].copy()
    fig = px.scatter(bs_plot, x='dot_ball_pct', y='wicket_pct',
                      size='balls_bowled', color='success_index',
                      hover_name='Bowler_Name', color_continuous_scale='RdYlGn',
                      title='Bowler: Dot Ball % vs Wicket % (bubble=deliveries, colour=success)', labels={'dot_ball_pct':'Dot Ball %','wicket_pct':'Wicket %'})
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=360)
    charts['bowler_scatter'] = _fig_to_json(fig)

    # Over-by-over run rate
    df['over_int'] = df['oversActual'].astype(int)
    over_stats = df.groupby('over_int').agg(
        avg_run=('run','mean'), bdry=('isBoundary','mean')
    ).reset_index()
    over_stats['bdry'] *= 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=['Avg Runs/Delivery','Boundary %'])
    fig.add_trace(go.Scatter(x=over_stats['over_int'], y=over_stats['avg_run'],
                              mode='lines+markers', name='Avg Runs',
                              line=dict(color='royalblue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=over_stats['over_int'], y=over_stats['bdry'],
                              mode='lines', name='Boundary %',
                              line=dict(color='tomato')), row=2, col=1)
    for row in [1, 2]:
        fig.add_vrect(x0=1, x1=6, fillcolor='lightgreen', opacity=0.12,
                       layer='below', line_width=0, row=row, col=1)
        fig.add_vrect(x0=16, x1=20, fillcolor='lightsalmon', opacity=0.15,
                       layer='below', line_width=0, row=row, col=1)
    fig.update_layout(title='Over-by-Over Trends', height=480,
                       margin=dict(l=10,r=10,t=40,b=10), xaxis2_title='Over')
    charts['over_trend'] = _fig_to_json(fig)

    return render_template('insights.html', charts=charts)


# ── API routes ────────────────────────────────────────────────────────────────

@app.route('/api/players')
def api_players():
    return jsonify({'batsmen': all_batsman_names(), 'bowlers': all_bowler_names()})


@app.route('/api/batsman/<path:name>')
def api_batsman(name):
    summary = bat_analytics.pregame_summary(name.strip())
    if 'error' in summary:
        return jsonify(summary), 404
    return jsonify(summary)


@app.route('/api/bowler/<path:name>')
def api_bowler(name):
    summary = bowl_analytics.pregame_summary(name.strip())
    if 'error' in summary:
        return jsonify(summary), 404
    return jsonify(summary)


@app.errorhandler(404)
def not_found(e):
    # Inline HTML — no template file needed, never crashes
    msg  = str(e)
    html = (
        '<!DOCTYPE html><html><head><title>404</title>'
        '<style>'
        'body{background:#0f1117;color:#e8eaf6;font-family:sans-serif;text-align:center;padding:4rem}'
        'h1{font-size:5rem;color:#e74c3c}'
        'a{color:#4f8ef7}'
        '</style></head><body>'
        '<h1>404</h1><p>' + msg + '</p><a href=/>&larr; Home</a>'
        '</body></html>'
    )
    return html, 404


@app.route('/favicon.ico')
def favicon():
    # Return empty 204 to stop browsers requesting favicon.ico repeatedly
    return '', 204


if __name__ == '__main__':
    app.run(debug=True, port=5000)
