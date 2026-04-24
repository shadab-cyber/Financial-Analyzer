"""
msme_routes.py
──────────────
Flask Blueprint for the MSME Analyzer feature.
All business logic lives in msme_analyzer.py.

HOW TO REGISTER IN app.py
──────────────────────────
After existing imports:
    from msme_routes import msme_bp

After app = Flask(__name__):
    app.register_blueprint(msme_bp)
"""

import os
import logging

from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename

from msme_analyzer import (
    run_gst_analysis,
    run_bank_analysis,
    run_combined_analysis,
)

logger      = logging.getLogger(__name__)
msme_bp     = Blueprint("msme", __name__)
UPLOAD_MSME = "uploads_msme"
os.makedirs(UPLOAD_MSME, exist_ok=True)


def _save(file) -> str:
    path = os.path.join(UPLOAD_MSME, secure_filename(file.filename))
    file.save(path)
    return path


def _remove(path: str):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ── Page ───────────────────────────────────────────────────────────────────────

@msme_bp.route("/msme-analyzer")
def msme_analyzer_page():
    is_android = request.args.get("app") == "android"
    return render_template("msme_analyzer.html", is_android=is_android)


# ── GST Upload ─────────────────────────────────────────────────────────────────

@msme_bp.route("/msme/upload-gst", methods=["POST"])
def msme_upload_gst():
    """
    Accept one or more GSTR-3B PDF files downloaded from gst.gov.in.
    Returns full GST summary dict.
    """
    files = request.files.getlist("files")
    if not files or not any(f.filename for f in files):
        return jsonify({"error": "No files uploaded."}), 400

    saved = []
    for f in files:
        if f and f.filename.lower().endswith(".pdf"):
            saved.append(_save(f))

    if not saved:
        return jsonify({
            "error": "Please upload GSTR-3B PDF files (.pdf format) downloaded from gst.gov.in."
        }), 400

    try:
        result = run_gst_analysis(saved)
        if "error" in result:
            return jsonify(result), 400
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.exception("msme_upload_gst error")
        return jsonify({"error": str(e)}), 500
    finally:
        for p in saved:
            _remove(p)


# ── Bank Upload ────────────────────────────────────────────────────────────────

@msme_bp.route("/msme/upload-bank", methods=["POST"])
def msme_upload_bank():
    """
    Accept bank statement as Excel (.xlsx/.xls), CSV, or PDF.
    PDF can be digital (pdfplumber) or scanned (Google Vision OCR).
    Returns full bank analytics dict.
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "No file uploaded."}), 400

    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ("xlsx", "xls", "csv", "pdf"):
        return jsonify({
            "error": f"Unsupported file type: .{ext}. Upload Excel (.xlsx), CSV, or PDF."
        }), 400

    path = _save(f)
    try:
        result = run_bank_analysis(path)
        if "error" in result:
            return jsonify(result), 400
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.exception("msme_upload_bank error")
        return jsonify({"error": f"Bank analysis failed: {e}"}), 500
    finally:
        _remove(path)


# ── Combined Report ────────────────────────────────────────────────────────────

@msme_bp.route("/msme/combined", methods=["POST"])
def msme_combined():
    """
    Accept previously computed gst + bank dicts from frontend state.
    Returns combined cross-analysis metrics and insights.
    """
    body = request.get_json() or {}
    gst  = body.get("gst")
    bank = body.get("bank")

    if not gst or not bank:
        return jsonify({"error": "Both 'gst' and 'bank' data objects are required."}), 400

    try:
        result = run_combined_analysis(gst, bank)
        return jsonify(result)
    except Exception as e:
        logger.exception("msme_combined error")
        return jsonify({"error": str(e)}), 500
        
