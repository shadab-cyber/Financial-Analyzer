from flask import Flask, request, jsonify, render_template, session
import os
import time
import logging
import threading
import uuid
import json
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

from Financial_Modelling import (
    run_historical_fs, run_ratio_analysis, run_common_size_statement,
    run_forecasting, run_fcff, run_wacc, run_terminal_value_dcf, run_scenario_analysis
)
from Technical_Analysis import run_technical_analysis
from Portfolio_Management import run_portfolio_analysis, fetch_price_for_symbol
from Performance_Analytics import run_complete_performance_analysis
from Strategy_Optimization import run_complete_optimization, run_backtest
from dcf_valuation import run_dcf_from_pdf_text
from pdf_text_extractor import extract_financials_from_pdf
from financialanalyzer import analyze_excel

# =============================================================================
# APP SETUP
# =============================================================================
app = Flask(__name__)

# ✅ FIX 1: Secret key from environment variable — never hardcoded
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)

# ✅ FIX 2: File upload size limit — 50MB max (prevents server crash attacks)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB — supports 5–8 annual report PDFs

# ✅ FIX 3: Production logging — errors go to a file, not stdout
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)

# Upload folders
UPLOAD_FOLDER_ANALYZER = 'uploads'
UPLOAD_FOLDER_DCF      = 'uploads_dcf'
ALLOWED_EXTENSIONS_PDF  = {'pdf'}
ALLOWED_EXTENSIONS_EXCEL = {'xlsx', 'xls'}

os.makedirs(UPLOAD_FOLDER_ANALYZER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_DCF,      exist_ok=True)

app.config['UPLOAD_FOLDER_ANALYZER'] = UPLOAD_FOLDER_ANALYZER
app.config['UPLOAD_FOLDER_DCF']      = UPLOAD_FOLDER_DCF

# =============================================================================
# JOB STORE — background processing
# =============================================================================
_jobs: dict = {}
_jobs_lock = threading.Lock()
JOBS_FOLDER = 'jobs'
os.makedirs(JOBS_FOLDER, exist_ok=True)

def _set_job(job_id: str, data: dict):
    with _jobs_lock:
        _jobs[job_id] = data
    try:
        with open(os.path.join(JOBS_FOLDER, f'{job_id}.json'), 'w') as f:
            json.dump(data, f)
    except Exception:
        pass

def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        if job_id in _jobs:
            return _jobs[job_id]
    try:
        path = os.path.join(JOBS_FOLDER, f'{job_id}.json')
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return None


# =============================================================================
# ANALYZER API  —  Screener.in Excel upload
# =============================================================================

def _run_excel_job(job_id: str, filepath: str):
    """Background thread: parse Excel and store result."""
    try:
        _set_job(job_id, {'status': 'processing', 'progress': 'Reading Excel file...', 'percent': 30})
        result = analyze_excel(filepath)
        _set_job(job_id, {'status': 'processing', 'progress': 'Computing ratios...', 'percent': 80})
        _set_job(job_id, {'status': 'done', 'result': result, 'percent': 100})
    except Exception as e:
        import traceback
        app.logger.error(f'[job:{job_id}] error: {e}\n{traceback.format_exc()}')
        _set_job(job_id, {'status': 'error', 'error': str(e)})
    finally:
        try:
            os.remove(filepath)
        except Exception:
            pass


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Accepts a single Screener.in Excel file (.xlsx).
    Returns job_id immediately. Browser polls /analyze/status/<job_id>.
    """
    try:
        file = request.files.get('files') or request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No file uploaded.'}), 400

        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in ('xlsx', 'xls'):
            return jsonify({'error': f'Please upload a Screener.in Excel file (.xlsx). Got: .{ext}'}), 400

        path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
        job_id = str(uuid.uuid4())
        _set_job(job_id, {'status': 'queued', 'progress': 'Upload received...', 'percent': 10})

        thread = threading.Thread(target=_run_excel_job, args=(job_id, path), daemon=True)
        thread.start()

        return jsonify({'job_id': job_id}), 202

    except Exception as e:
        import traceback
        app.logger.error(f'analyze error: {e}\n{traceback.format_exc()}')
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/analyze/status/<job_id>', methods=['GET'])
def analyze_status(job_id):
    """Poll every 2 seconds for job result."""
    job = _get_job(job_id)
    if job is None:
        return jsonify({'status': 'error', 'error': 'Job not found or expired.'}), 404
    return jsonify(job)

@app.route('/debug-pdf', methods=['GET'])
def debug_pdf_page():
    """Browser-friendly debug page — upload PDFs and see exactly what gets extracted."""
    return """<!DOCTYPE html>
<html><head><title>PDF Debug</title>
<style>
body{font-family:monospace;padding:20px;background:#1e1e1e;color:#d4d4d4}
h2{color:#569cd6}textarea{width:100%;height:200px;background:#252526;color:#ce9178;border:1px solid #555;padding:8px;font-size:13px}
button{background:#0e639c;color:#fff;border:none;padding:10px 24px;font-size:14px;cursor:pointer;margin-top:8px}
button:hover{background:#1177bb}
#out{white-space:pre-wrap;background:#252526;color:#9cdcfe;padding:16px;margin-top:16px;border:1px solid #555;max-height:80vh;overflow-y:auto;font-size:12px}
.match{color:#4ec9b0;font-weight:bold}.empty{color:#f44747}label{color:#9cdcfe}
</style></head>
<body>
<h2>🔍 PDF Debug — See exactly what gets extracted</h2>
<form id="f" enctype="multipart/form-data">
  <label>Upload your financial statement PDFs:</label><br><br>
  <input type="file" name="files" multiple accept=".pdf" style="color:#fff"><br>
  <button type="submit">Analyze & Show Raw Extraction</button>
</form>
<div id="out">Results will appear here...</div>
<script>
document.getElementById('f').addEventListener('submit', async e => {
  e.preventDefault();
  const out = document.getElementById('out');
  out.textContent = 'Extracting... please wait';
  const fd = new FormData(e.target);
  try {
    const r = await fetch('/debug-pdf-run', {method:'POST', body:fd});
    const txt = await r.text();
    // Highlight matched lines
    out.textContent = txt;

  } catch(err) {
    out.textContent = 'Error: ' + err.message;
  }
});
</script>
</body></html>""", 200, {'Content-Type': 'text/html'}


@app.route('/debug-pdf-run', methods=['POST'])
def debug_pdf_run():
    """Runs extraction and returns plain text report."""
    paths = []
    out = []
    try:
        files = request.files.getlist('files')
        if not files or not any(f.filename for f in files):
            return 'No files uploaded', 400

        for file in files:
            if not file or not allowed_pdf(file.filename):
                continue
            path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
            paths.append(path)

            # Show what each strategy extracts individually
            out.append(f"{'='*70}")
            out.append(f"FILE: {file.filename}")
            out.append(f"{'='*70}")

            # Raw PyMuPDF text
            try:
                import fitz
                doc = fitz.open(path)
                out.append(f"\nPyMuPDF pages: {len(doc)}")
                out.append("--- PyMuPDF RAW TEXT (first 100 lines) ---")
                raw_lines = []
                for pg in doc:
                    t = pg.get_text("text")
                    for ln in t.splitlines():
                        ln = ln.strip()
                        if ln:
                            raw_lines.append(ln)
                for i, ln in enumerate(raw_lines[:100]):
                    out.append(f"  [{i:03d}] {ln}")
                out.append(f"  ... total PyMuPDF lines: {len(raw_lines)}")
                doc.close()
            except Exception as e:
                out.append(f"PyMuPDF error: {e}")

            # pdfplumber text
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    out.append(f"\npdfplumber pages: {len(pdf.pages)}")
                    out.append("--- pdfplumber RAW TEXT (first 50 lines) ---")
                    plumb_lines = []
                    for pg in pdf.pages:
                        t = pg.extract_text()
                        if t:
                            for ln in t.splitlines():
                                ln = ln.strip()
                                if ln:
                                    plumb_lines.append(ln)
                    for i, ln in enumerate(plumb_lines[:50]):
                        out.append(f"  [{i:03d}] {ln}")
                    out.append(f"  ... total pdfplumber lines: {len(plumb_lines)}")
            except Exception as e:
                out.append(f"pdfplumber error: {e}")

            # Full extraction result
            lines = pdf_to_text(path)
            out.append(f"\n--- COMBINED EXTRACTION: {len(lines)} lines ---")
            for i, ln in enumerate(lines[:200]):
                out.append(f"  [{i:03d}] {ln}")

            out.append("")
            out.append("--- INCOME STATEMENT EXTRACTION ---")
            inc = extract_income_series(lines)
            for k, v in inc.items():
                tag = "[MATCHED]" if v else ""
                out.append(f"  {tag} {k}: {v}")

            out.append("")
            out.append("--- BALANCE SHEET EXTRACTION ---")
            bal = extract_balance_series(lines)
            for k, v in bal.items():
                tag = "[MATCHED]" if v else ""
                out.append(f"  {tag} {k}: {v}")

            out.append("")
            out.append("--- CASH FLOW EXTRACTION ---")
            cf = extract_cashflow_series(lines)
            for k, v in cf.items():
                tag = "[MATCHED]" if v else ""
                out.append(f"  {tag} {k}: {v}")

            out.append("")
            matched_inc = [k for k,v in inc.items() if v]
            matched_bal = [k for k,v in bal.items() if v]
            matched_cf  = [k for k,v in cf.items()  if v]
            out.append(f"SUMMARY:")
            out.append(f"  Income  fields matched: {matched_inc}")
            out.append(f"  Balance fields matched: {matched_bal}")
            out.append(f"  CashFlow fields matched:{matched_cf}")

        return "\n".join(out), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        import traceback
        return f"Error: {e}\n{traceback.format_exc()}", 500
    finally:
        for p in paths:
            cleanup(p)


# =============================================================================
# DCF API
# =============================================================================
@app.route('/dcf/upload', methods=['POST'])
def dcf_from_pdf():
    paths = []
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No PDF files uploaded'}), 400

        files = request.files.getlist('files')
        all_text_blocks = []

        for file in files:
            if not file or not allowed_pdf(file.filename):
                continue
            path = save_temp_file(file, UPLOAD_FOLDER_DCF)
            paths.append(path)

            extracted   = extract_financials_from_pdf(path)
            text_blocks = extracted['structured_financials']['raw_text']
            all_text_blocks.extend(text_blocks)

        if not all_text_blocks:
            return jsonify({'error': 'Could not extract text from the uploaded PDFs. '
                            'Please ensure the files are readable annual reports.'}), 400

        dcf_result = run_dcf_from_pdf_text(all_text_blocks)
        return jsonify(dcf_result)

    except Exception as e:
        app.logger.error(f'dcf error: {e}')
        return err('DCF calculation failed', 500)
    finally:
        for p in paths:
            cleanup(p)


# =============================================================================
# FINANCIAL MODELLING APIs  (shared helper to remove repeated code)
# =============================================================================
def _excel_upload_route(run_fn, extra_params=None):
    """Generic handler for all Excel-upload modelling routes."""
    path = None
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No Excel file uploaded'}), 400

        file = request.files['file']
        if not file.filename.lower().endswith('.xlsx'):
            return jsonify({'error': 'Please upload a valid .xlsx Excel file'}), 400

        path   = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
        kwargs = extra_params or {}
        result = run_fn(path, **kwargs)
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'{run_fn.__name__} error: {e}')
        return err(f'{run_fn.__name__} failed', 500)
    finally:
        cleanup(path)


@app.route('/financial-modelling/upload', methods=['POST'])
def financial_modelling():
    return _excel_upload_route(run_historical_fs)

@app.route('/financial-modelling/ratio-analysis/upload', methods=['POST'])
def ratio_analysis():
    return _excel_upload_route(run_ratio_analysis)

@app.route('/financial-modelling/common-size/upload', methods=['POST'])
def common_size_statement():
    return _excel_upload_route(run_common_size_statement)

@app.route('/financial-modelling/forecasting/upload', methods=['POST'])
def forecasting():
    years = int(request.form.get('forecast_years', 5))
    return _excel_upload_route(run_forecasting, {'forecast_years': years})

@app.route('/financial-modelling/fcff/upload', methods=['POST'])
def fcff():
    return _excel_upload_route(run_fcff)

@app.route('/financial-modelling/wacc/upload', methods=['POST'])
def wacc():
    return _excel_upload_route(run_wacc)

@app.route('/financial-modelling/terminal-value/upload', methods=['POST'])
def terminal_value():
    return _excel_upload_route(run_terminal_value_dcf)

@app.route('/financial-modelling/scenario-analysis/upload', methods=['POST'])
def scenario_analysis():
    years = int(request.form.get('forecast_years', 5))
    return _excel_upload_route(run_scenario_analysis, {'forecast_years': years})


# =============================================================================
# TECHNICAL ANALYSIS API
# =============================================================================
@app.route('/technical-analysis/analyze', methods=['POST'])
def technical_analysis():
    try:
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Stock symbol is required'}), 400

        symbol    = data.get('symbol', '').strip()
        timeframe = data.get('timeframe', '1d')
        period    = data.get('period', '1y')

        if not symbol:
            return jsonify({'error': 'Symbol cannot be empty'}), 400

        result = run_technical_analysis(symbol, timeframe, period)
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'technical_analysis error: {e}')
        return err('Technical analysis failed', 500)


# =============================================================================
# PORTFOLIO MANAGEMENT API
# =============================================================================
@app.route('/portfolio-management/analyze', methods=['POST'])
def portfolio_management():
    try:
        data     = request.get_json()
        holdings = data.get('holdings', []) if data else []

        if not holdings:
            return jsonify({'error': 'At least one holding is required'}), 400

        result = run_portfolio_analysis(holdings)

        if result.get('success'):
            session['portfolio_data'] = {
                'holdings':        holdings,
                'analysis_result': result,
                'timestamp':       datetime.now().isoformat()
            }

        return jsonify(result)

    except Exception as e:
        app.logger.error(f'portfolio_management error: {e}')
        return err('Portfolio analysis failed', 500)


@app.route('/portfolio-management/get-data', methods=['GET'])
def get_portfolio_data():
    try:
        portfolio_data = session.get('portfolio_data')
        if not portfolio_data:
            return jsonify({'error': 'No portfolio data found. '
                            'Please analyse your portfolio first.'}), 404
        return jsonify(portfolio_data)
    except Exception as e:
        app.logger.error(f'get_portfolio_data error: {e}')
        return err('Could not retrieve portfolio data', 500)


# =============================================================================
# PERFORMANCE ANALYTICS API
# =============================================================================
@app.route('/performance-analytics/analyze', methods=['POST'])
def performance_analytics():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        holdings_history = data.get('holdings_history', [])
        if not holdings_history:
            return jsonify({'error': 'Holdings history is required'}), 400

        result = run_complete_performance_analysis(
            holdings_history,
            data.get('holdings_detail', []),
            data.get('cash_flows', []),
            data.get('trades'),
            data.get('signals'),
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'performance_analytics error: {e}')
        return err('Performance analysis failed', 500)


# =============================================================================
# STRATEGY OPTIMIZATION APIs
# =============================================================================
@app.route('/strategy-optimization/complete', methods=['POST'])
def strategy_optimization_complete():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symbol     = data.get('symbol')
        start_date = data.get('start_date')
        end_date   = data.get('end_date')

        if not all([symbol, start_date, end_date]):
            return jsonify({'error': 'symbol, start_date and end_date are required'}), 400

        result = run_complete_optimization(
            symbol, data.get('strategy_config', {}), start_date, end_date
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'strategy_optimization error: {e}')
        return err('Strategy optimization failed', 500)


@app.route('/strategy-optimization/backtest', methods=['POST'])
def strategy_optimization_backtest():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symbol     = data.get('symbol')
        start_date = data.get('start_date')
        end_date   = data.get('end_date')

        if not all([symbol, start_date, end_date]):
            return jsonify({'error': 'symbol, start_date and end_date are required'}), 400

        result = run_backtest(
            symbol,
            data.get('strategy_config', {}),
            start_date, end_date,
            data.get('initial_capital', 100000)
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'backtest error: {e}')
        return err('Backtest failed', 500)

from flask import send_from_directory

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files like sample Excel"""
    try:
        return send_from_directory('static', filename)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# =============================================================================
# ✅ FIX 7: Production server start — debug=False, use gunicorn in real deploy
# =============================================================================
if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port       = int(os.environ.get('PORT', 5000))

    print('=' * 55)
    print(' 💼 Financial Analyzer — Starting Server')
    print('=' * 55)
    print(f' Mode  : {"DEVELOPMENT" if debug_mode else "PRODUCTION"}')
    print(f' Port  : {port}')
    print(f' Debug : {debug_mode}')
    print('=' * 55)
    print(' Routes:')
    print(f'   http://127.0.0.1:{port}/')
    print(f'   http://127.0.0.1:{port}/dcf')
    print(f'   http://127.0.0.1:{port}/financial-modelling')
    print(f'   http://127.0.0.1:{port}/technical-analysis')
    print(f'   http://127.0.0.1:{port}/portfolio-management')
    print(f'   http://127.0.0.1:{port}/performance-analytics')
    print(f'   http://127.0.0.1:{port}/strategy-optimization')
    print('=' * 55)

    app.run(debug=debug_mode, host='0.0.0.0', port=port)
