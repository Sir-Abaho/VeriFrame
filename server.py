"""
server.py — AI Video Detector API Server
==========================================
Run this file to start the app:
    python server.py

Then open your browser to:
    http://localhost:5000
"""

import os
import uuid
import threading
from flask import Flask, request, jsonify, send_from_directory
from detector import analyze_video

app = Flask(__name__, static_folder=".")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Stores analysis progress and results keyed by job_id
jobs = {}

ALLOWED_EXTENSIONS = {
    "mp4", "mov", "avi", "mkv", "webm",
    "flv", "wmv", "m4v", "mpeg", "mpg", "3gp"
}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Supported: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save file with unique name
    ext      = file.filename.rsplit(".", 1)[1].lower()
    job_id   = str(uuid.uuid4())
    filename = f"{job_id}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Register job
    jobs[job_id] = {
        "status":   "processing",
        "step":     0,
        "total":    6,
        "message":  "Starting analysis...",
        "result":   None,
        "error":    None,
    }

    # Run analysis in background thread
    thread = threading.Thread(target=run_analysis, args=(job_id, filepath))
    thread.daemon = True
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    return jsonify(job)


@app.route("/result/<job_id>")
def result(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    if job["status"] != "done":
        return jsonify({"error": "Analysis not complete yet."}), 202
    return jsonify(job["result"])


# ─────────────────────────────────────────────
#  BACKGROUND ANALYSIS WORKER
# ─────────────────────────────────────────────

def run_analysis(job_id, filepath):
    def progress(step, total, message):
        jobs[job_id]["step"]    = step
        jobs[job_id]["total"]   = total
        jobs[job_id]["message"] = message

    try:
        result = analyze_video(filepath, progress_callback=progress)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
    finally:
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception:
            pass


# ─────────────────────────────────────────────
#  START
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AI Video Detector — Server Starting")
    print("  Open your browser to: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
