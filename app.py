"""
app.py
======
Flask web application for the MNIST Multi-Digit Handwritten Recognition System.

Routes:
  GET/POST  /signup     → register (SQLite user row + password hash)
  GET/POST  /login      → sign in (Flask session)
  GET/POST  /logout     → end session
  GET  /                → canvas (requires login)
  POST /predict         → JSON (requires login)
"""

import base64
import io
import json
import os
import random
from functools import wraps
from urllib.parse import urlparse

import cv2
import numpy as np
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from PIL import Image

from auth_store import create_user, db_path, init_db, verify_login
from predict import predict_textlines  # our OCR prediction helper


# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")

def get_db_path() -> str:
    return db_path(app)


@app.before_request
def _ensure_db() -> None:
    init_db(get_db_path())


def login_required_page(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)

    return wrapped


def login_required_api(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Authentication required. Please log in."}), 401
        return f(*args, **kwargs)

    return wrapped


def safe_next_url(param: str | None) -> str:
    if not param:
        return url_for("index")
    p = urlparse(param)
    if p.netloc or p.scheme:
        return url_for("index")
    if not p.path.startswith("/"):
        return url_for("index")
    return param


def build_output_analysis(result: dict) -> dict:
    """Human-readable breakdown for the browser (success or failure)."""
    err = result.get("error")
    lines = result.get("lines") or []
    prediction_text = (result.get("prediction_text") or "").strip()
    pred_lines = result.get("prediction_lines") or []
    engine = result.get("engine") or "unknown"

    items: list[dict] = []

    if err:
        items.append({"label": "Status", "value": "Could not produce a reading", "tone": "bad"})
        items.append({"label": "Details", "value": err, "tone": "neutral"})
        low = err.lower()
        if "no characters" in low or "no text lines" in low or "not loaded" in low:
            items.append(
                {
                    "label": "Suggestion",
                    "value": "Draw darker, thicker strokes on the white canvas; keep one clear text line.",
                    "tone": "tip",
                }
            )
        elif "ctc" in low or "blank" in low:
            items.append(
                {
                    "label": "Suggestion",
                    "value": "Add spacing between symbols; simple block letters often work best.",
                    "tone": "tip",
                }
            )
        else:
            items.append(
                {
                    "label": "Suggestion",
                    "value": "Try clearing the canvas, write larger, then Predict again.",
                    "tone": "tip",
                }
            )
    else:
        display = prediction_text if prediction_text else "(empty)"
        items.append({"label": "Recognized text", "value": display, "tone": "good"})
        items.append({"label": "Lines detected", "value": str(len(pred_lines)), "tone": "neutral"})
        if engine == "char_roi":
            items.append(
                {
                    "label": "Pipeline",
                    "value": "Character ROIs + EMNIST-style CNN (per symbol).",
                    "tone": "neutral",
                }
            )
        elif engine == "ctc_crnn":
            items.append(
                {
                    "label": "Pipeline",
                    "value": "Whole text line + CRNN and CTC decode.",
                    "tone": "neutral",
                }
            )

    if lines and not err:
        confs = [float(li.get("confidence", 0)) for li in lines if li.get("confidence") is not None]
        avg = sum(confs) / len(confs) if confs else 0.0
        if avg >= 70:
            verdict = "Model scores this as fairly confident."
        elif avg >= 40:
            verdict = "Mixed confidence — verify the text if it matters."
        else:
            verdict = "Low confidence — redraw more clearly."
        items.append(
            {
                "label": "Average confidence",
                "value": f"{avg:.1f}% — {verdict}",
                "tone": "neutral",
            }
        )

    return {"engine": engine, "items": items}


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"):
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username", "")
        p1 = request.form.get("password", "")
        p2 = request.form.get("password2", "")
        if p1 != p2:
            return render_template(
                "signup.html",
                error="Passwords do not match.",
                username=username,
            ), 400
        err = create_user(get_db_path(), username, p1)
        if err:
            return render_template("signup.html", error=err, username=username), 400
        flash("Account created. You can log in now.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))

    next_raw = request.args.get("next") or ""
    if request.method == "POST":
        next_raw = request.form.get("next") or next_raw
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        uid, err = verify_login(get_db_path(), username, password)
        if err:
            return (
                render_template(
                    "login.html",
                    error=err,
                    username=username,
                    next_url=next_raw,
                ),
                401,
            )
        session["user_id"] = uid
        session["username"] = username.strip().lower()
        return redirect(safe_next_url(next_raw))

    return render_template("login.html", next_url=next_raw)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/preview")
def output_preview():
    """Static layout sample (no inference)."""
    return render_template("output_preview.html")


@app.route("/live-demo")
def live_demo():
    """
    Run the real OCR stack on a server-generated image and show output in the browser.
    No login required — for quick demos.
    """
    h = w = 280
    img = np.zeros((h, w), dtype=np.uint8)
    label = random.choice(["42", "Hi", "A", "8", "x1"])
    cv2.putText(
        img,
        label,
        (24, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.1,
        255,
        9,
        cv2.LINE_AA,
    )
    vis = cv2.bitwise_not(img)
    _, buf = cv2.imencode(".png", vis)
    image_data_url = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    result = predict_textlines(img.copy())
    result = dict(result)
    result["analysis"] = build_output_analysis(result)
    raw_json = json.dumps(result, indent=2, ensure_ascii=False)
    return render_template(
        "live_demo.html",
        result=result,
        image_data_url=image_data_url,
        raw_json=raw_json,
    )


# ─────────────────────────────────────────────
# ROUTE: HOME
# ─────────────────────────────────────────────


@app.route("/")
@login_required_page
def index():
    """Serve the main drawing canvas page."""
    return render_template("index.html")


# ─────────────────────────────────────────────
# ROUTE: PREDICT
# ─────────────────────────────────────────────


@app.route("/predict", methods=["POST"])
@login_required_api
def predict():
    """
    Receive a Base64-encoded canvas PNG, preprocess it,
    run multi-digit prediction, and return JSON results.

    Expected JSON body:
        { "image": "data:image/png;base64,<data>" }

    Response JSON:
        {
          "prediction": "583",
          "digits": [
            {"digit": 5, "confidence": 97.3},
            {"digit": 8, "confidence": 95.1},
            {"digit": 3, "confidence": 98.7}
          ],
          "average_confidence": 97.0
        }
    """
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data received."}), 400

    try:
        # ── 1. Decode Base64 image ──────────────────────────────────────────
        image_data = data["image"]
        # Strip the data-URL prefix: "data:image/png;base64,<actual_data>"
        if "," in image_data:
            image_data = image_data.split(",")[1]

        raw_bytes = base64.b64decode(image_data)

        # ── 2. Open with Pillow & convert to grayscale ──────────────────────
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("L")

        # ── 3. Convert to NumPy array ────────────────────────────────────────
        img_array = np.array(pil_img, dtype=np.uint8)

        # ── 4. Invert colors so digits are WHITE on BLACK (MNIST style) ──────
        #       Canvas: black digits on white background
        #       MNIST:  white digits on black background
        img_array = cv2.bitwise_not(img_array)

        # ── 5. Predict (handles multi-line OCR with CRNN+CTC) ────────────────
        result = predict_textlines(img_array)
        result = dict(result)
        result["analysis"] = build_output_analysis(result)
        if result.get("error"):
            return jsonify(result), 500
        return jsonify(result)

    except Exception as e:
        err_payload = {"error": str(e), "prediction_lines": [], "prediction_text": "", "lines": []}
        err_payload["analysis"] = build_output_analysis(err_payload)
        return jsonify(err_payload), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\nMNIST Digit Recognition Server (Flask)")
    print("Open your browser at http://127.0.0.1:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
