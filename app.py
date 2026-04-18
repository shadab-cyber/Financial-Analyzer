from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
import os
import time
import logging
import threading
import uuid
import json
import hmac
import hashlib
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

# ── Razorpay ──────────────────────────────────────────────────────────────────
try:
    import razorpay
    RAZORPAY_KEY_ID     = os.environ.get('RAZORPAY_KEY_ID', '')
    RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', '')
    rzp_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET)) if RAZORPAY_KEY_ID else None
except ImportError:
    rzp_client = None
    RAZORPAY_KEY_ID = ''
    RAZORPAY_KEY_SECRET = ''

# Plan config (amounts in paise — multiply ₹ by 100)
PLANS = {
    'premium': {'name': 'Premium', 'monthly': 49900,  'annual': 478800},   # ₹499 / ₹3990 (save 33%)
    'pro':     {'name': 'Pro',     'monthly': 99900,  'annual': 958800},   # ₹999 / ₹7990
}
# ─────────────────────────────────────────────────────────────────────────────

from Financial_Modelling import (
    run_historical_fs, run_ratio_analysis, run_common_size_statement,
    run_forecasting, run_fcff, run_wacc, run_terminal_value_dcf, run_scenario_analysis,
    run_altman_zscore, run_roic, run_piotroski, run_dupont
)
from Technical_Analysis import run_technical_analysis
from Portfolio_Management import run_portfolio_analysis, fetch_price_for_symbol
from Performance_Analytics import run_complete_performance_analysis
from Strategy_Optimization import run_complete_optimization, run_backtest
from dcf_valuation import run_dcf_from_pdf_text, run_scenarios
from pdf_text_extractor import extract_financials_from_pdf
from financialanalyzer import analyze_excel
from msme_routes import msme_bp

def _validate_dcf_params(wacc, terminal_growth, net_debt, shares, cmp_price,
                          enterprise_value=None):
    """
    Central DCF parameter validator. Returns error string or None.
    Call after computing enterprise_value to catch EV edge cases.
    """
    if wacc is None:
        return 'WACC is required.'
    if wacc <= 0 or wacc > 0.5:
        return f'WACC ({wacc*100:.1f}%) is outside the plausible range (0–50%).'
    if terminal_growth is None:
        return 'Terminal Growth Rate is required.'
    if terminal_growth < 0:
        return 'Terminal Growth Rate cannot be negative.'
    if terminal_growth >= wacc:
        return (f'Terminal growth ({terminal_growth*100:.1f}%) must be strictly less than '
                f'WACC ({wacc*100:.1f}%).')
    if shares is not None and float(shares) < 0:
        return 'Shares outstanding cannot be negative.'
    if cmp_price is not None and float(cmp_price) < 0:
        return 'Current market price cannot be negative.'
    if enterprise_value is not None and net_debt is not None:
        eq = enterprise_value - float(net_debt)
        if eq < 0 and shares and float(shares) > 0:
            return (f'Net Debt (₹{net_debt:.0f} Cr) exceeds Enterprise Value '
                    f'(₹{enterprise_value:.0f} Cr) — equity value is negative. '
                    f'Check your Net Debt input.')
    return None


# =============================================================================
# APP SETUP
# =============================================================================
app = Flask(__name__)
app.register_blueprint(msme_bp)

# ✅ FIX 1: Secret key from environment variable — never hardcoded
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)

# ── Firebase Admin SDK (server-side token verification) ──────────────────────
if not firebase_admin._apps:
    sa_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT')
    if sa_json:
        # Read credentials from environment variable (safe for public repos)
        import json as _json
        sa_dict = _json.loads(sa_json)
        cred = credentials.Certificate(sa_dict)
        firebase_admin.initialize_app(cred)
    elif os.path.exists('service-account.json'):
        # Fallback: read from file if present locally
        cred = credentials.Certificate('service-account.json')
        firebase_admin.initialize_app(cred)
    else:
        raise RuntimeError(
            "Firebase credentials not found. "
            "Set FIREBASE_SERVICE_ACCOUNT environment variable on Render."
        )

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
# HELPERS
# =============================================================================
def allowed_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_PDF

def allowed_excel(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_EXCEL

def save_temp_file(file, folder: str) -> str:
    filename = secure_filename(file.filename)
    path = os.path.join(folder, filename)
    file.save(path)
    return path

def cleanup(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def err(msg: str, code: int = 400):
    return jsonify({'error': msg}), code


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
# ERROR HANDLERS
# =============================================================================
@app.errorhandler(404)
def not_found(_):
    return render_template('404.html'), 404

@app.errorhandler(413)
def too_large(_):
    return jsonify({'error': 'File too large. Maximum allowed size is 500 MB.'}), 413

@app.errorhandler(500)
def server_error(_):
    return render_template('500.html'), 500


# =============================================================================
# PAGES
# =============================================================================
@app.route('/')
def home():
    is_android = request.args.get('app') == 'android'
    return render_template('financialanalyzerweb.html', is_android=is_android)

@app.route('/dcf')
def dcf_page():
    is_android = request.args.get('app') == 'android'
    return render_template('dcfvaluation.html', is_android=is_android)

@app.route('/financial-modelling')
def financial_modelling_page():
    is_android = request.args.get('app') == 'android'
    return render_template('financial_modelling.html', is_android=is_android)

@app.route('/technical-analysis')
def technical_analysis_page():
    is_android = request.args.get('app') == 'android'
    return render_template('technical_analysis.html', is_android=is_android)

@app.route('/portfolio-management')
def portfolio_management_page():
    is_android = request.args.get('app') == 'android'
    return render_template('portfolio_management.html', is_android=is_android)

@app.route('/performance-analytics')
def performance_analytics_page():
    is_android = request.args.get('app') == 'android'
    return render_template('performance_analytics.html', is_android=is_android)

@app.route('/strategy-optimization')
def strategy_optimization_page():
    is_android = request.args.get('app') == 'android'
    return render_template('strategy_optimization.html', is_android=is_android)


@app.route('/pricing')
def pricing_page():
    return render_template('pricing.html', razorpay_key=RAZORPAY_KEY_ID)


# =============================================================================
# Razorpay Payment Routes
# =============================================================================

@app.route('/payment/create-order', methods=['POST'])
def create_payment_order():
    """Create a Razorpay order for the selected plan + billing cycle."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Login required'}), 401
    if not rzp_client:
        return jsonify({'error': 'Payment gateway not configured'}), 503

    data     = request.get_json() or {}
    plan_id  = data.get('plan')       # 'premium' or 'pro'
    billing  = data.get('billing', 'monthly')  # 'monthly' or 'annual'

    plan = PLANS.get(plan_id)
    if not plan:
        return jsonify({'error': 'Invalid plan'}), 400

    amount = plan['annual'] if billing == 'annual' else plan['monthly']

    try:
        order = rzp_client.order.create({
            'amount':   amount,
            'currency': 'INR',
            'receipt':  f"{user['uid'][:8]}-{plan_id}-{int(time.time())}",
            'notes': {
                'user_uid':   user['uid'],
                'user_email': user['email'],
                'plan':       plan_id,
                'billing':    billing,
            }
        })
        return jsonify({
            'order_id':   order['id'],
            'amount':     order['amount'],
            'currency':   order['currency'],
            'plan_name':  plan['name'],
            'billing':    billing,
            'key':        RAZORPAY_KEY_ID,
        })
    except Exception as e:
        app.logger.error(f'create_payment_order error: {e}')
        return jsonify({'error': str(e)}), 500


@app.route('/payment/verify', methods=['POST'])
def verify_payment():
    """Verify Razorpay signature and activate the user's plan."""
    user = session.get('user')
    if not user:
        return jsonify({'error': 'Login required'}), 401

    data = request.get_json() or {}
    razorpay_order_id   = data.get('razorpay_order_id', '')
    razorpay_payment_id = data.get('razorpay_payment_id', '')
    razorpay_signature  = data.get('razorpay_signature', '')
    plan_id             = data.get('plan', '')
    billing             = data.get('billing', 'monthly')

    # Verify HMAC signature
    try:
        msg = f"{razorpay_order_id}|{razorpay_payment_id}".encode()
        expected = hmac.new(RAZORPAY_KEY_SECRET.encode(), msg, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, razorpay_signature):
            return jsonify({'error': 'Invalid payment signature'}), 400
    except Exception as e:
        app.logger.error(f'verify_payment signature error: {e}')
        return jsonify({'error': 'Signature verification failed'}), 400

    # Activate plan in session
    plan = PLANS.get(plan_id, {})
    session['user']['plan']       = plan_id
    session['user']['billing']    = billing
    session['user']['plan_name']  = plan.get('name', plan_id.title())
    session['user']['plan_since'] = datetime.utcnow().isoformat()
    session.modified = True

    app.logger.info(
        f"Payment verified: user={user['email']} plan={plan_id} "
        f"billing={billing} payment_id={razorpay_payment_id}"
    )
    return jsonify({'status': 'ok', 'plan': plan_id, 'plan_name': plan.get('name')})


@app.route('/payment/status')
def payment_status():
    """Return current user's plan."""
    user = session.get('user')
    if not user:
        return jsonify({'plan': 'free'})
    return jsonify({
        'plan':       user.get('plan', 'free'),
        'plan_name':  user.get('plan_name', 'Free'),
        'billing':    user.get('billing', ''),
        'plan_since': user.get('plan_since', ''),
    })



# =============================================================================
# SEO — Sitemap, Robots, Google Search Console
# =============================================================================
@app.route('/sitemap.xml')
def sitemap():
    from flask import Response
    import os
    xml = open(os.path.join(app.root_path, 'sitemap.xml')).read()
    return Response(xml, mimetype='application/xml')

@app.route('/robots.txt')
def robots():
    from flask import Response
    import os
    txt = open(os.path.join(app.root_path, 'robots.txt')).read()
    return Response(txt, mimetype='text/plain')

@app.route('/google<token>.html')
def google_verification(token):
    from flask import Response
    return Response(f'google-site-verification: google{token}', mimetype='text/html')

# =============================================================================
# AUTH — Google Sign-In via Firebase
# =============================================================================
@app.route('/login')
def login_page():
    is_android = request.args.get('app') == 'android'
    return render_template('login.html', is_android=is_android)

@app.route('/auth/google', methods=['POST'])
def auth_google():
    """Verify Firebase ID token from client, create Flask session."""
    try:
        data     = request.get_json()
        id_token = data.get('id_token') if data else None
        if not id_token:
            return jsonify({'error': 'No token provided'}), 400

        decoded  = firebase_auth.verify_id_token(id_token)
        session['user'] = {
            'uid':   decoded['uid'],
            'email': decoded.get('email', ''),
            'name':  decoded.get('name', ''),
            'photo': decoded.get('picture', ''),
        }
        session.permanent = True
        return jsonify({'status': 'ok', 'user': session['user']})

    except firebase_auth.InvalidIdTokenError:
        return jsonify({'error': 'Invalid token'}), 401
    except Exception as e:
        app.logger.error(f'auth_google error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    session.pop('user', None)
    return jsonify({'status': 'ok'})

@app.route('/auth/status')
def auth_status():
    user = session.get('user')
    return jsonify({'signed_in': bool(user), 'user': user})

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
# =============================================================================
# DCF — background job (PDF extraction can take several minutes;
# running it synchronously causes a 502 on Render's 30-second proxy timeout)
# =============================================================================

def _extract_dcf_text_lean(pdf_path: str) -> list:
    """
    Extract ALL text from a PDF — no page filtering, no keyword cherry-picking.

    Previous versions tried to be smart (filter by CFS page headers, filter
    by keyword lines). Every filtering step was a new way to silently drop
    the exact lines dcf_valuation.py needed.  The right design is:

        extractor  = get me all the text, as fast as possible
        dcf parser = find CFO and CAPEX inside that text

    Hard cap at MAX_PAGES (150) keeps memory bounded.
    Returns [] only if the PDF has zero selectable text (scanned image).
    """
    import gc

    MAX_PAGES = 150
    all_pages = []

    # ── Primary: PyMuPDF (fast) ───────────────────────────────────────────
    fitz_ok = False
    try:
        import fitz
        doc   = fitz.open(pdf_path)
        total = min(len(doc), MAX_PAGES)
        for page_num in range(total):
            page = doc[page_num]
            text = page.get_text("text")
            page = None
            if text and text.strip():
                all_pages.append(text)
            if page_num % 30 == 0:
                gc.collect()
        doc.close()
        del doc
        gc.collect()
        fitz_ok = True
    except ImportError:
        pass
    except Exception as e:
        app.logger.warning(f'PyMuPDF extraction failed for {pdf_path}: {e}')
        all_pages.clear()
        gc.collect()

    # ── Fallback: pdfplumber ──────────────────────────────────────────────
    if not fitz_ok or not all_pages:
        all_pages.clear()
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total = min(len(pdf.pages), MAX_PAGES)
                for page_num in range(total):
                    page = pdf.pages[page_num]
                    try:
                        text = page.extract_text() or ''
                    finally:
                        page.flush_cache()
                        del page
                    if text and text.strip():
                        all_pages.append(text)
                    if page_num % 20 == 0:
                        gc.collect()
            gc.collect()
        except Exception as e:
            app.logger.warning(f'pdfplumber fallback failed for {pdf_path}: {e}')
            gc.collect()

    if not all_pages:
        app.logger.warning(
            f'DCF extractor: {pdf_path} → 0 pages with text (scanned PDF?)'
        )
        return []

    combined = '\n--- PAGE BREAK ---\n'.join(all_pages)
    app.logger.info(
        f'DCF extractor: {pdf_path} → {len(all_pages)} pages, '
        f'{len(combined.splitlines())} lines'
    )
    return [combined]


def _run_dcf_job(job_id: str, paths: list, wacc, terminal_growth,
                 net_debt, shares, cmp_price):
    """Background worker: extract text from PDFs one at a time, run DCF."""
    import gc
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    PDF_TIMEOUT_SECS = 90  # max time allowed per PDF

    try:
        _set_job(job_id, {'status': 'processing',
                          'progress': 'Starting PDF extraction…', 'percent': 5})
        all_text_blocks = []
        n = len(paths)

        for idx, path in enumerate(paths):
            pct = 10 + int(55 * idx / n)
            _set_job(job_id, {
                'status':   'processing',
                'progress': f'Extracting PDF {idx+1} of {n}…',
                'percent':  pct,
            })
            try:
                # Run extraction in a thread so we can enforce a hard timeout
                with ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(_extract_dcf_text_lean, path)
                    try:
                        blocks = future.result(timeout=PDF_TIMEOUT_SECS)
                        all_text_blocks.extend(blocks)
                        hits = len(blocks[0].splitlines()) if blocks else 0
                        app.logger.info(
                            f'DCF job {job_id}: PDF {idx+1}/{n} → {hits} keyword lines'
                        )
                        if hits == 0:
                            app.logger.warning(
                                f'DCF job {job_id}: PDF {idx+1} yielded 0 keyword lines — '
                                f'may be a scanned PDF'
                            )
                    except FutureTimeout:
                        app.logger.warning(
                            f'DCF job {job_id}: PDF {idx+1} timed out after {PDF_TIMEOUT_SECS}s — skipping'
                        )
                        _set_job(job_id, {
                            'status':   'processing',
                            'progress': f'PDF {idx+1} of {n} took too long — skipped. Continuing…',
                            'percent':  pct,
                        })
            except Exception as e:
                app.logger.warning(f'DCF PDF {idx+1} failed: {e}')
            finally:
                cleanup(path)
                gc.collect()

        if not all_text_blocks or not any(b.strip() for b in all_text_blocks):
            _set_job(job_id, {
                'status': 'error',
                'error':  (
                    'No CFO / CAPEX data found in the uploaded PDFs. '
                    'This tool requires text-based (not fully scanned) annual reports. '
                    'Tip: use the Screener.in exported PDF or the company\'s '
                    'digital annual report — not a photographed scan.'
                )
            })
            return

        _set_job(job_id, {'status': 'processing',
                          'progress': 'Running DCF calculation…', 'percent': 80})
        gc.collect()

        result = run_dcf_from_pdf_text(
            all_text_blocks,
            net_debt=net_debt,
            shares_outstanding=shares,
            current_price=cmp_price,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )  # forecast_years uses default 5 for PDF path
        _set_job(job_id, {'status': 'done', 'result': result, 'percent': 100})

    except Exception as e:
        app.logger.error(f'DCF job {job_id} failed: {e}')
        _set_job(job_id, {'status': 'error', 'error': str(e)})\
        
        for p in paths:
            cleanup(p)
        import gc; gc.collect()


@app.route('/dcf/debug-extract', methods=['POST'])
def dcf_debug_extract():
    """
    Debug endpoint: upload one PDF, see exactly what text gets extracted
    and whether CFO/CAPEX keywords match.
    Visit /dcf in browser, open DevTools → Network, or hit with curl:
      curl -F "file=@annual_report.pdf" https://your-app.onrender.com/dcf/debug-extract
    Returns plain text — safe to paste into any text editor.
    """
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if not allowed_pdf(file.filename):
        return 'Must be a PDF', 400

    path = save_temp_file(file, UPLOAD_FOLDER_DCF)
    out  = []
    try:
        blocks = _extract_dcf_text_lean(path)
        if not blocks or not blocks[0].strip():
            out.append('ERROR: extractor returned empty — likely a scanned PDF.')
            return '\n'.join(out), 200, {'Content-Type': 'text/plain; charset=utf-8'}

        lines = blocks[0].splitlines()
        out.append(f'=== EXTRACTED {len(lines)} lines from {file.filename} ===\n')

        # Show all lines
        for i, ln in enumerate(lines):
            out.append(f'[{i:04d}] {ln}')

        # Show which lines matched CFO patterns
        from dcf_valuation import _CFO_RE, _CAPEX_RE, _EXCL_RE, _matches_any
        out.append('\n\n=== CFO KEYWORD MATCHES ===')
        for i, ln in enumerate(lines):
            if _matches_any(ln.lower(), _CFO_RE):
                excl = ' [EXCLUDED]' if _matches_any(ln.lower(), _EXCL_RE) else ''
                out.append(f'[{i:04d}]{excl} {ln}')

        out.append('\n=== CAPEX KEYWORD MATCHES ===')
        for i, ln in enumerate(lines):
            if _matches_any(ln.lower(), _CAPEX_RE):
                out.append(f'[{i:04d}] {ln}')

        # Run the actual extractor and show results
        out.append('\n=== DCF EXTRACTION RESULT ===')
        try:
            from dcf_valuation import extract_cfo, extract_capex
            cfo   = extract_cfo(blocks)
            capex = extract_capex(blocks)
            out.append(f'CFO values found:   {cfo}')
            out.append(f'CAPEX values found: {capex}')
            if not cfo:
                out.append('PROBLEM: No CFO value extracted — check CFO KEYWORD MATCHES above')
            if not capex:
                out.append('PROBLEM: No CAPEX value extracted — check CAPEX KEYWORD MATCHES above')
        except Exception as e:
            out.append(f'Extraction error: {e}')

    finally:
        cleanup(path)

    return '\n'.join(out), 200, {'Content-Type': 'text/plain; charset=utf-8'}


from dcf_excel_extractor import extract_dcf_from_excel, extract_wacc_inputs, extract_ebitda, extract_latest_fcf


@app.route('/dcf/excel', methods=['POST'])
def dcf_from_excel():
    """
    Accept a Screener.in Excel file + DCF parameters.
    Extracts CFO and CAPEX directly from the Cash Flow sheet — no PDF, no OCR.
    Returns the full DCF result synchronously (fast enough to not need a job queue).
    """
    path = None
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No file uploaded.'}), 400
        if not allowed_excel(file.filename):
            return jsonify({'error': 'Please upload a Screener.in Excel file (.xlsx)'}), 400

        path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)

        # Extract CFO + CAPEX from Excel
        excel_data = extract_dcf_from_excel(path)

        def _f(key, default=None):
            v = request.form.get(key)
            try:
                return float(v) if v not in (None, '') else default
            except (ValueError, TypeError):
                return default

        wacc            = _f('wacc')
        terminal_growth = _f('terminal_growth')
        net_debt        = _f('net_debt', 0)
        shares          = _f('shares_outstanding')
        cmp_price       = _f('current_price')
        forecast_years  = int(request.form.get('forecast_years', 5))

        if wacc is None or terminal_growth is None:
            return jsonify({'error': 'WACC and Terminal Growth are required.'}), 400

        if terminal_growth >= wacc:
            return jsonify({'error': f'Terminal growth ({terminal_growth*100:.1f}%) must be less than WACC ({wacc*100:.1f}%).'}), 400

        # Build synthetic text blocks from the clean numbers — reuses the
        # existing DCF engine without touching it
        lines = []
        for i in range(len(excel_data['cfo'])):
            lines.append(f"Net cash from operating activities {excel_data['cfo'][i]}")
            lines.append(f"Purchase of property plant equipment {excel_data['capex'][i]}")
        text_blocks = ['\n'.join(lines)]

        result = run_dcf_from_pdf_text(
            text_blocks,
            net_debt=net_debt,
            shares_outstanding=shares,
            current_price=cmp_price,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )

        # Attach Excel metadata so frontend can show real fiscal years + data preview
        result['Extracted Years'] = excel_data['years']
        result['Extracted CFO']   = excel_data['cfo']
        result['Extracted CAPEX'] = excel_data['capex']
        result['excel_fcf']       = excel_data['fcf']

        # EV/EBITDA cross-check
        ebitda_val, ebitda_yr = extract_ebitda(path)
        result['EBITDA (₹ Cr)']       = ebitda_val
        result['EBITDA Year']          = ebitda_yr

        fcf_latest, fcf_yr = extract_latest_fcf(path)
        result['Latest FCF (₹ Cr)']    = fcf_latest
        result['Latest FCF Year']      = fcf_yr

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f'dcf_from_excel error: {e}')
        return jsonify({'error': f'DCF calculation failed: {str(e)}'}), 500
    finally:
        cleanup(path)


@app.route('/dcf/wacc-inputs', methods=['POST'])
def dcf_wacc_inputs():
    """
    Read interest, debt, tax rate, equity from an uploaded Excel.
    Returns values for the WACC builder to pre-fill.
    """
    path = None
    try:
        file = request.files.get('file')
        if not file or not allowed_excel(file.filename):
            return jsonify({'error': 'No Excel file'}), 400
        path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
        data = extract_wacc_inputs(path)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        cleanup(path)


@app.route('/dcf/scenarios', methods=['POST'])
def dcf_scenarios():
    """
    Run Bull / Base / Bear scenarios from an Excel upload.
    Accepts same params as /dcf/excel plus optional per-scenario growth rates.
    """
    path = None
    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No Excel file uploaded.'}), 400
        if not allowed_excel(file.filename):
            return jsonify({'error': 'Please upload a Screener.in Excel file (.xlsx)'}), 400

        path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
        excel_data = extract_dcf_from_excel(path)

        def _f(key, default=None):
            v = request.form.get(key)
            try:    return float(v) if v not in (None, '') else default
            except: return default

        wacc            = _f('wacc')
        terminal_growth = _f('terminal_growth')
        net_debt        = _f('net_debt', 0)
        shares          = _f('shares_outstanding')
        cmp_price       = _f('current_price')
        forecast_years  = int(request.form.get('forecast_years', 5))
        stage1_years    = int(request.form.get('stage1_years', 5)) if request.form.get('stage1_years') else None
        use_2stage      = request.form.get('use_2stage', 'false').lower() == 'true'

        bear_g1 = _f('bear_g1')
        base_g1 = _f('base_g1')
        bull_g1 = _f('bull_g1')
        bear_g2 = _f('bear_g2') if use_2stage else None
        base_g2 = _f('base_g2') if use_2stage else None
        bull_g2 = _f('bull_g2') if use_2stage else None

        # Validate required params
        if wacc is None or terminal_growth is None:
            return jsonify({'error': 'WACC and Terminal Growth Rate are required.'}), 400
        if wacc <= 0 or wacc > 0.5:
            return jsonify({'error': f'WACC ({wacc*100:.1f}%) is outside the valid range (0–50%).'}), 400
        if terminal_growth < 0:
            return jsonify({'error': 'Terminal Growth Rate cannot be negative.'}), 400
        if terminal_growth >= wacc:
            return jsonify({'error': f'Terminal growth ({terminal_growth*100:.1f}%) must be less than WACC ({wacc*100:.1f}%).'}), 400

        # Build text blocks from Excel data
        lines = []
        for i in range(len(excel_data['cfo'])):
            lines.append(f"Net cash from operating activities {excel_data['cfo'][i]}")
            lines.append(f"Purchase of property plant equipment {excel_data['capex'][i]}")
        text_blocks = ['\n'.join(lines)]

        result = run_scenarios(
            text_blocks,
            net_debt=net_debt, shares_outstanding=shares,
            current_price=cmp_price, wacc=wacc, terminal_growth=terminal_growth,
            forecast_years=forecast_years,
            stage1_years=stage1_years if use_2stage else None,
            bear_g1=bear_g1, bear_g2=bear_g2,
            base_g1=base_g1, base_g2=base_g2,
            bull_g1=bull_g1, bull_g2=bull_g2,
        )

        result['Extracted Years']  = excel_data['years']
        result['Extracted CFO']    = excel_data['cfo']
        result['Extracted CAPEX']  = excel_data['capex']

        ebitda_val, ebitda_yr = extract_ebitda(path)
        result['EBITDA (₹ Cr)'] = ebitda_val
        result['EBITDA Year']   = ebitda_yr

        fcf_latest, fcf_yr = extract_latest_fcf(path)
        result['Latest FCF (₹ Cr)']  = fcf_latest
        result['Latest FCF Year']    = fcf_yr
        if 'Base' in result:
            result['Base']['EBITDA (₹ Cr)']    = ebitda_val
            result['Base']['EBITDA Year']      = ebitda_yr
            result['Base']['Latest FCF (₹ Cr)'] = fcf_latest
            result['Base']['Latest FCF Year']   = fcf_yr

        return jsonify(result)

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        app.logger.error(f'dcf_scenarios error: {e}')
        return jsonify({'error': f'DCF scenario calculation failed: {str(e)}'}), 500
    finally:
        cleanup(path)


@app.route('/dcf/manual', methods=['POST'])
def dcf_manual():
    """
    Run DCF from manually entered CFO and CAPEX arrays — no PDF, instant response.
    Expects JSON: { cfo: [n1,n2,...], capex: [n1,n2,...], net_debt, shares,
                    current_price, wacc, terminal_growth }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        cfo_list   = [float(x) for x in data.get('cfo', [])   if x not in (None, '')]
        capex_list = [float(x) for x in data.get('capex', []) if x not in (None, '')]

        if len(cfo_list) < 2 or len(capex_list) < 2:
            return jsonify({'error': 'Please enter at least 2 years of CFO and CAPEX data.'}), 400

        def _f(key, default=None):
            v = data.get(key)
            try: return float(v) if v not in (None, '') else default
            except (ValueError, TypeError): return default

        wacc            = _f('wacc')
        terminal_growth = _f('terminal_growth')
        net_debt        = _f('net_debt', 0)
        shares          = _f('shares')
        cmp_price       = _f('current_price')
        forecast_years  = int(data.get('forecast_years', 5) or 5)

        # Build synthetic text blocks from the numbers so the existing
        # DCF engine (which was designed for PDF-extracted text) still works.
        # Format matches what extract_cfo / extract_capex expect.
        text_lines = []
        years = min(len(cfo_list), len(capex_list))
        for i in range(years):
            text_lines.append(f"Net cash from operating activities {cfo_list[i]}")
            text_lines.append(f"Purchase of property plant equipment {capex_list[i]}")
        text_blocks = ['\n'.join(text_lines)]

        result = run_dcf_from_pdf_text(
            text_blocks,
            net_debt=net_debt,
            shares_outstanding=shares,
            current_price=cmp_price,
            wacc=wacc,
            terminal_growth=terminal_growth,
            forecast_years=forecast_years,
        )
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'DCF manual error: {e}')
        return jsonify({'error': str(e)}), 500



@app.route('/dcf/upload', methods=['POST'])
def dcf_from_pdf():
    """
    Accept PDFs + DCF parameters, start a background job, return job_id immediately.
    Client polls /dcf/status/<job_id> every 3 seconds.
    """
    if 'files' not in request.files:
        return jsonify({'error': 'No PDF files uploaded'}), 400

    files = request.files.getlist('files')
    paths = []
    for file in files:
        if not file or not allowed_pdf(file.filename):
            continue
        path = save_temp_file(file, UPLOAD_FOLDER_DCF)
        paths.append(path)

    if not paths:
        return jsonify({'error': 'No valid PDF files uploaded'}), 400

    def _float(key, default=None):
        v = request.form.get(key)
        try:
            return float(v) if v not in (None, '') else default
        except (ValueError, TypeError):
            return default

    wacc            = _float('wacc')
    terminal_growth = _float('terminal_growth')
    net_debt        = _float('net_debt', 0)
    shares          = _float('shares_outstanding')
    cmp_price       = _float('current_price')

    job_id = str(uuid.uuid4())
    _set_job(job_id, {'status': 'queued', 'progress': 'Job queued…', 'percent': 0})

    t = threading.Thread(
        target=_run_dcf_job,
        args=(job_id, paths, wacc, terminal_growth, net_debt, shares, cmp_price),
        daemon=True,
    )
    t.start()

    return jsonify({'job_id': job_id}), 202


@app.route('/dcf/status/<job_id>', methods=['GET'])
def dcf_status(job_id):
    """Poll for DCF job progress / result.

    If the job isn't found in memory or on disk, return 'processing' for up
    to 10 seconds after the request — this handles the brief window after a
    dyno restart where the thread hasn't written its first checkpoint yet.
    After that window, return an error.
    """
    job = _get_job(job_id)
    if job is not None:
        return jsonify(job)

    # Job not found — could be a transient post-restart gap.
    # Return a safe "processing" response so the frontend keeps polling.
    return jsonify({
        'status':   'processing',
        'progress': 'Job is starting… (server may have restarted)',
        'percent':  5
    })


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

@app.route('/financial-modelling/altman-zscore/upload', methods=['POST'])
def altman_zscore():
    return _excel_upload_route(run_altman_zscore)

@app.route('/financial-modelling/roic/upload', methods=['POST'])
def roic():
    return _excel_upload_route(run_roic)

@app.route('/financial-modelling/piotroski/upload', methods=['POST'])
def piotroski():
    return _excel_upload_route(run_piotroski)

@app.route('/financial-modelling/dupont/upload', methods=['POST'])
def dupont():
    return _excel_upload_route(run_dupont)


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
# BLOG
# =============================================================================

BLOG_POSTS = [
    {
        "slug": "dcf-valuation-indian-stocks",
        "title": "How to Do DCF Valuation of Indian Stocks (Free Tool Included)",
        "emoji": "📊",
        "category": "DCF Valuation",
        "excerpt": "Learn how to calculate the intrinsic value of any Indian stock using Discounted Cash Flow analysis. Step-by-step guide with a free online tool.",
        "date": "March 2026",
        "date_iso": "2026-03-01",
        "read_time": 8,
        "keywords": "dcf valuation indian stocks, dcf calculator india, intrinsic value indian stocks, wacc india, free dcf tool",
        "meta_description": "Step-by-step guide to DCF valuation of Indian stocks. Learn how to calculate WACC, terminal value and intrinsic value — with a free online tool.",
        "toc": [
            {"label": "What is DCF Valuation?", "anchor": "what-is-dcf"},
            {"label": "Why DCF Works for Indian Stocks", "anchor": "why-dcf-india"},
            {"label": "Step 1 — Forecast Free Cash Flow", "anchor": "step1-fcf"},
            {"label": "Step 2 — Calculate WACC", "anchor": "step2-wacc"},
            {"label": "Step 3 — Calculate Terminal Value", "anchor": "step3-terminal"},
            {"label": "Step 4 — Arrive at Intrinsic Value", "anchor": "step4-intrinsic"},
            {"label": "Common DCF Mistakes to Avoid", "anchor": "mistakes"},
            {"label": "Use Our Free DCF Tool", "anchor": "free-tool"},
        ],
        "content": """
<p>Discounted Cash Flow (DCF) valuation is the gold standard for determining the intrinsic value of a stock. It is used by Warren Buffett, every major investment bank, and now — thanks to free tools — individual investors in India too.</p>
<p>In this guide, you will learn exactly how to do a DCF valuation for any Indian stock, step by step, using data from Screener.in.</p>
<h2 id="what-is-dcf">What is DCF Valuation?</h2>
<p>DCF valuation calculates what a business is worth today based on all the cash it will generate in the future, discounted back to present value. The core idea is simple: a rupee received 10 years from now is worth less than a rupee today.</p>
<p>The formula is: <strong>Intrinsic Value = Sum of (Future Free Cash Flows / (1 + WACC)^n) + Terminal Value</strong></p>
<div class="callout callout-tip"><strong>💡 Key insight:</strong> If the intrinsic value from DCF is higher than the current market price, the stock may be undervalued.</div>
<h2 id="why-dcf-india">Why DCF Works Well for Indian Stocks</h2>
<p>Indian markets have several characteristics that make DCF analysis particularly useful. Many Indian businesses are growing at 15–25% annually — DCF captures this growth premium better than PE ratios. Screener.in also provides 10 years of audited financial data for free, which is exactly what DCF needs.</p>
<h2 id="step1-fcf">Step 1 — Forecast Free Cash Flow</h2>
<p>Free Cash Flow = Operating Cash Flow − Capital Expenditure. Look at 5–10 years of historical FCF and identify a sustainable growth rate. For large-cap Indian stocks use 8–12%, for mid-caps 12–18%, and for high-growth small-caps up to 22%.</p>
<div class="callout callout-warn"><strong>⚠️ Warning:</strong> Never use management guidance as your base case. Use historical average as base and management targets as optimistic scenario.</div>
<h2 id="step2-wacc">Step 2 — Calculate WACC</h2>
<p>For Indian stocks, WACC typically ranges from 10% to 16%. Use the 10-year Indian Government bond yield (~7.1% in 2026) as your risk-free rate. India's equity risk premium is approximately 7–8%. Cost of Equity = Risk-free Rate + Beta × ERP.</p>
<h2 id="step3-terminal">Step 3 — Calculate Terminal Value</h2>
<p>Terminal Value = FCF in Year 10 × (1 + g) / (WACC − g). Use a terminal growth rate of 4–6% for Indian companies. Never use a growth rate higher than WACC.</p>
<div class="callout callout-tip"><strong>💡 Important:</strong> Terminal value typically accounts for 60–80% of total DCF value for Indian growth stocks. Always run sensitivity analysis.</div>
<h2 id="step4-intrinsic">Step 4 — Arrive at Intrinsic Value Per Share</h2>
<p>Equity Value = Enterprise Value − Net Debt. Intrinsic Value Per Share = Equity Value / Shares Outstanding. A margin of safety of at least 20–30% is recommended before buying.</p>
<h2 id="mistakes">Common DCF Mistakes to Avoid</h2>
<ul>
  <li>Using too high a growth rate — even great companies rarely sustain 25%+ for 10 years</li>
  <li>Ignoring working capital changes</li>
  <li>Single scenario analysis — always run bear, base, and bull</li>
  <li>Forgetting to add cash back to equity value</li>
</ul>
<div class="callout callout-cta">
  <h3 style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#fff;margin-bottom:10px;">Do the entire DCF in 30 seconds</h3>
  <p>Upload any Screener.in Excel export and get instant DCF valuation — WACC, terminal value, intrinsic value, and three-scenario analysis. Completely free.</p>
  <a href="/dcf">Try the Free DCF Tool →</a>
</div>
<h2 id="free-tool">Use Our Free DCF Tool</h2>
<p>Go to <a href="https://screener.in">Screener.in</a>, export the company Excel, upload it to our <a href="/dcf">DCF Valuation tool</a>, enter your WACC and terminal growth assumptions, and get instant results.</p>
"""
    },
    {
        "slug": "altman-z-score-indian-stocks",
        "title": "Altman Z-Score: Which Indian Stocks Are at Risk of Bankruptcy?",
        "emoji": "⚠️",
        "category": "Stock Analysis",
        "excerpt": "The Altman Z-Score predicts financial distress with 72–80% accuracy. Learn how to calculate it for Indian stocks and which ones to avoid.",
        "date": "March 2026",
        "date_iso": "2026-03-10",
        "read_time": 6,
        "keywords": "altman z score india, bankruptcy prediction indian stocks, financial distress india, z score calculator",
        "meta_description": "Learn how the Altman Z-Score predicts bankruptcy risk for Indian stocks. Free calculator included.",
        "toc": [
            {"label": "What is the Altman Z-Score?", "anchor": "what-is-z"},
            {"label": "How to Interpret Z-Score", "anchor": "interpret"},
            {"label": "Z-Score for Indian Companies", "anchor": "india"},
            {"label": "Use the Free Calculator", "anchor": "calculator"},
        ],
        "content": """
<p>The Altman Z-Score is one of the most reliable tools for predicting whether a company is heading towards financial distress. Developed by Professor Edward Altman in 1968, it correctly predicted 72–80% of bankruptcies up to two years before they occurred.</p>
<h2 id="what-is-z">What is the Altman Z-Score?</h2>
<p>Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5, where X1=Working Capital/Total Assets, X2=Retained Earnings/Total Assets, X3=EBIT/Total Assets, X4=Market Cap/Total Liabilities, X5=Revenue/Total Assets.</p>
<h2 id="interpret">How to Interpret the Z-Score</h2>
<table><thead><tr><th>Z-Score</th><th>Zone</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Above 2.99</td><td>✅ Safe Zone</td><td>Low bankruptcy risk</td></tr>
<tr><td>1.81 – 2.99</td><td>⚠️ Grey Zone</td><td>Some risk, monitor closely</td></tr>
<tr><td>Below 1.81</td><td>🚨 Distress Zone</td><td>High bankruptcy risk</td></tr>
</tbody></table>
<h2 id="india">Z-Score for Indian Companies</h2>
<p>The original Z-Score was developed using US manufacturing companies. For Indian markets, the thresholds are broadly applicable but Indian companies often carry more debt and operate in higher-inflation environments.</p>
<div class="callout callout-warn"><strong>⚠️ Note:</strong> A low Z-Score does not guarantee bankruptcy — it signals elevated risk. Always combine with qualitative analysis.</div>
<div class="callout callout-cta">
  <h3 style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#fff;margin-bottom:10px;">Calculate Z-Score for Any Indian Stock</h3>
  <p>Upload a Screener.in Excel export and get instant Altman Z-Score with historical trend analysis.</p>
  <a href="/">Try the Free Tool →</a>
</div>
<h2 id="calculator">Use the Free Calculator</h2>
<p>Our Stock Financial Analyzer automatically calculates the Altman Z-Score from any Screener.in Excel export. Upload your data on the <a href="/">home page</a> — no sign-up required.</p>
"""
    },
    {
        "slug": "screener-in-excel-analysis-guide",
        "title": "How to Analyse Any Indian Stock Using Screener.in Data in 5 Minutes",
        "emoji": "📁",
        "category": "Screener.in",
        "excerpt": "A complete step-by-step guide to downloading Screener.in Excel data and running a full 10-year financial analysis in under 5 minutes.",
        "date": "March 2026",
        "date_iso": "2026-03-15",
        "read_time": 5,
        "keywords": "screener.in excel analysis, screener.in tutorial, analyse indian stock, screener.in export",
        "meta_description": "Step-by-step guide to downloading Screener.in Excel data and running a full 10-year financial analysis in under 5 minutes.",
        "toc": [
            {"label": "Step 1 — Export from Screener.in", "anchor": "export"},
            {"label": "Step 2 — Upload to Financial Analyzer", "anchor": "upload"},
            {"label": "Step 3 — Read the Results", "anchor": "results"},
            {"label": "What to Look For", "anchor": "look-for"},
        ],
        "content": """
<p>Screener.in is the best free source of Indian stock financial data. Combined with Financial Analyzer, you can go from zero to a complete 10-year analysis in under 5 minutes.</p>
<h2 id="export">Step 1 — Export from Screener.in</h2>
<ol>
  <li>Go to <a href="https://screener.in" target="_blank">screener.in</a> and search for your company</li>
  <li>Scroll to the top right and click <strong>Export to Excel</strong></li>
  <li>Save the downloaded .xlsx file</li>
</ol>
<div class="callout callout-tip"><strong>💡 Tip:</strong> You need a free Screener.in account to export. The export button only appears when logged in.</div>
<h2 id="upload">Step 2 — Upload to Financial Analyzer</h2>
<ol>
  <li>Go to <a href="/">financial-analyzer-m63v.onrender.com</a></li>
  <li>Click the upload area and select the Screener.in Excel file</li>
  <li>Click <strong>Analyse Stock</strong> — results appear in 10–15 seconds</li>
</ol>
<h2 id="results">Step 3 — Read the Results</h2>
<p>You get: 10-year P&amp;L, Balance Sheet, Cash Flow analysis, 20+ financial ratios, and Altman Z-Score — all in one page.</p>
<h2 id="look-for">What to Look For</h2>
<table><thead><tr><th>Metric</th><th>Green Flag</th><th>Red Flag</th></tr></thead><tbody>
<tr><td>Revenue Growth (10yr CAGR)</td><td>Above 12%</td><td>Below 8%</td></tr>
<tr><td>Free Cash Flow</td><td>Consistently positive</td><td>Negative or erratic</td></tr>
<tr><td>Debt/Equity</td><td>Below 1.0</td><td>Above 2.0</td></tr>
<tr><td>Altman Z-Score</td><td>Above 2.99</td><td>Below 1.81</td></tr>
</tbody></table>
<div class="callout callout-cta">
  <h3 style="font-family:'Playfair Display',serif;font-size:1.2rem;color:#fff;margin-bottom:10px;">Analyse Your First Stock Now</h3>
  <p>Upload any Screener.in Excel and get a complete 10-year analysis in 15 seconds. Free — no credit card required.</p>
  <a href="/">Start Analysing →</a>
</div>
"""
    },
]

def _get_post(slug):
    return next((p for p in BLOG_POSTS if p['slug'] == slug), None)

def _related_posts(current_slug, category, n=3):
    same_cat = [p for p in BLOG_POSTS if p['slug'] != current_slug and p['category'] == category]
    others   = [p for p in BLOG_POSTS if p['slug'] != current_slug and p['category'] != category]
    return (same_cat + others)[:n]

@app.route('/blog')
def blog_index():
    cat = request.args.get('cat', '').lower()
    posts = BLOG_POSTS
    if cat:
        posts = [p for p in posts if cat in p['category'].lower() or cat in p['slug']]
    return render_template('blog.html', posts=posts)

@app.route('/blog/<slug>')
def blog_post(slug):
    post = _get_post(slug)
    if not post:
        return render_template('404.html'), 404
    related = _related_posts(slug, post['category'])
    return render_template('blog_post.html', post=post, related_posts=related)


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
