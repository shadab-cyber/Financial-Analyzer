from flask import Flask, request, jsonify, render_template, session
import os
import time
import logging
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
from financialanalyzer import (
    pdf_to_rows, pdf_to_text, detect_type, detect_all_types,
    extract_income_series, extract_balance_series, extract_cashflow_series,
    analyze_income, analyze_balance, analyze_cashflow
)

# =============================================================================
# APP SETUP
# =============================================================================
app = Flask(__name__)

# ✅ FIX 1: Secret key from environment variable — never hardcoded
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)

# ✅ FIX 2: File upload size limit — 50MB max (prevents server crash attacks)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

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

os.makedirs(UPLOAD_FOLDER_ANALYZER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_DCF,      exist_ok=True)

app.config['UPLOAD_FOLDER_ANALYZER'] = UPLOAD_FOLDER_ANALYZER
app.config['UPLOAD_FOLDER_DCF']      = UPLOAD_FOLDER_DCF


# =============================================================================
# HELPERS
# =============================================================================
def allowed_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_PDF


def save_temp_file(file, folder):
    """Save an uploaded file with a unique timestamped name. Returns the path."""
    os.makedirs(folder, exist_ok=True)
    filename = f"{int(time.time() * 1000)}_{secure_filename(file.filename)}"
    path = os.path.join(folder, filename)
    file.save(path)
    return path


def cleanup(path):
    """✅ FIX 4: Delete uploaded file after processing — prevents disk fill."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def err(msg, code=500):
    """✅ FIX 5: Safe error responses — no internal details leaked to user."""
    app.logger.error(msg)
    # Show a friendly message to the user; details go to app.log
    friendly = {
        400: "Invalid request. Please check your input and try again.",
        404: "Resource not found.",
        413: "File too large. Maximum allowed size is 50 MB.",
        500: "Something went wrong on our end. Please try again.",
    }
    return jsonify({"error": friendly.get(code, "An error occurred.")}), code


# =============================================================================
# ✅ FIX 6: Custom error pages — no raw Flask 404/500 pages in production
# =============================================================================
@app.errorhandler(404)
def not_found(_):
    return render_template('404.html'), 404


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large. Maximum allowed size is 50 MB."}), 413


@app.errorhandler(500)
def server_error(_):
    return render_template('500.html'), 500


# =============================================================================
# PAGES
# =============================================================================
@app.route('/')
def home():
    return render_template('financialanalyzerweb.html')

@app.route('/dcf')
def dcf_page():
    return render_template('dcfvaluation.html')

@app.route('/financial-modelling')
def financial_modelling_page():
    return render_template('financial_modelling.html')

@app.route('/technical-analysis')
def technical_analysis_page():
    return render_template('technical_analysis.html')

@app.route('/portfolio-management')
def portfolio_management_page():
    return render_template('portfolio_management.html')

@app.route('/performance-analytics')
def performance_analytics_page():
    return render_template('performance_analytics.html')

@app.route('/strategy-optimization')
def strategy_optimization_page():
    return render_template('strategy_optimization.html')


# =============================================================================
# ✅ PRICE PROXY — browser calls this; Flask calls yfinance; no CORS issues
# =============================================================================
@app.route('/get-price/<symbol>')
def get_price(symbol):
    try:
        # Basic sanitisation — only allow alphanumeric + common ticker chars
        safe = ''.join(c for c in symbol if c.isalnum() or c in ('-', '&', '.'))
        price, source = fetch_price_for_symbol(safe)
        if price:
            return jsonify({'symbol': safe, 'price': price, 'source': source})
        return jsonify({'symbol': safe, 'price': None, 'error': 'Price not available'}), 404
    except Exception as e:
        app.logger.error(f'get_price error for {symbol}: {e}')
        return jsonify({'error': 'Could not fetch price'}), 500


# =============================================================================
# ANALYZER API
# =============================================================================
@app.route('/analyze', methods=['POST'])
def analyze():
    paths = []
    try:
        files = request.files.getlist('files')
        income_data = balance_data = cash_data = None

        for file in files:
            if not file or not allowed_pdf(file.filename):
                continue
            path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
            paths.append(path)

            # Use pdfplumber row extraction (structured tables, much better than raw text)
            rows  = pdf_to_rows(path)
            lines = [" ".join(r) for r in rows]

            types_in_file = detect_all_types(lines)
            if not types_in_file:
                primary = detect_type(lines, file.filename)
                if primary:
                    types_in_file = [primary]

            app.logger.error(f'[analyze] {file.filename}: types={types_in_file} rows={len(rows)}')

            if 'income' in types_in_file and income_data is None:
                candidate = extract_income_series(rows)
                if any(v for v in candidate.values()):
                    income_data = candidate
                    app.logger.error(f'[analyze] income fields found: {[k for k,v in candidate.items() if v]}')

            if 'balance' in types_in_file and balance_data is None:
                candidate = extract_balance_series(rows)
                if any(v for v in candidate.values()):
                    balance_data = candidate
                    app.logger.error(f'[analyze] balance fields found: {[k for k,v in candidate.items() if v]}')

            if 'cash' in types_in_file and cash_data is None:
                candidate = extract_cashflow_series(rows)
                if any(v for v in candidate.values()):
                    cash_data = candidate
                    app.logger.error(f'[analyze] cash fields found: {[k for k,v in candidate.items() if v]}')

        result = {}
        if income_data and any(v for v in income_data.values()):
            try:
                result['Income Statement'] = analyze_income(income_data)
            except Exception as e:
                app.logger.error(f'analyze_income error: {e}')
                result['Income Statement'] = {'error': str(e)}

        if balance_data and any(v for v in balance_data.values()):
            try:
                result['Balance Sheet'] = analyze_balance(balance_data, income_data)
            except Exception as e:
                app.logger.error(f'analyze_balance error: {e}')
                result['Balance Sheet'] = {'error': str(e)}

        if cash_data and any(v for v in cash_data.values()):
            try:
                result['Cash Flow'] = analyze_cashflow(cash_data, balance_data, income_data)
            except Exception as e:
                app.logger.error(f'analyze_cashflow error: {e}')
                result['Cash Flow'] = {'error': str(e)}

        app.logger.error(f'[analyze] final sections: {list(result.keys())}')
        return jsonify(result)

    except Exception as e:
        app.logger.error(f'analyze error: {e}')
        return err('analyze failed', 500)
    finally:
        for p in paths:
            cleanup(p)


@app.route('/debug-analyze', methods=['POST'])
def debug_analyze():
    """Debug endpoint: upload PDFs here to see exactly what rows are extracted
    and which keywords matched. Returns plain text you can read in the browser.
    Use: curl -F 'files=@yourfile.pdf' https://your-app.onrender.com/debug-analyze
    """
    paths = []
    out = []
    try:
        files = request.files.getlist('files')
        for file in files:
            if not file or not allowed_pdf(file.filename):
                continue
            path = save_temp_file(file, UPLOAD_FOLDER_ANALYZER)
            paths.append(path)

            rows  = pdf_to_rows(path)
            lines = [" ".join(r) for r in rows]

            out.append(f"\n{'='*70}")
            out.append(f"FILE: {file.filename}  ({len(rows)} rows extracted)")
            out.append(f"{'='*70}")
            out.append("\n--- FIRST 100 ROWS ---")
            for i, row in enumerate(rows[:100]):
                label = row[0][:90] if row else ''
                nums  = [c for c in row[1:] if c.strip()] if len(row)>1 else []
                out.append(f"  [{i:03d}] {label!r:92s}  extra_cells={nums}")

            out.append("\n--- DETECTED TYPES ---")
            out.append(f"  detect_all_types : {detect_all_types(lines)}")
            out.append(f"  detect_type      : {detect_type(lines, file.filename)}")

            out.append("\n--- INCOME EXTRACTION RESULT ---")
            inc = extract_income_series(rows)
            for k, v in inc.items():
                out.append(f"  {k}: {v}")

            out.append("\n--- BALANCE EXTRACTION RESULT ---")
            bal = extract_balance_series(rows)
            for k, v in bal.items():
                out.append(f"  {k}: {v}")

            out.append("\n--- CASHFLOW EXTRACTION RESULT ---")
            cf = extract_cashflow_series(rows)
            for k, v in cf.items():
                out.append(f"  {k}: {v}")

        return "\n".join(out), 200, {'Content-Type': 'text/plain; charset=utf-8'}
    except Exception as e:
        import traceback
        return f"Debug error: {e}\n{traceback.format_exc()}", 500
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
