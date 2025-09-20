from flask import Flask, request, jsonify
from tasks import process_receipt_task
import uuid
import os
import json

app = Flask(__name__)

BASE_DIR = "/var/am/tasks"

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    session = str(uuid.uuid4())
    session_dir = os.path.join(BASE_DIR, session)
    os.makedirs(session_dir, exist_ok=True)

    image_path = os.path.join(session_dir, "0.jpeg")
    file.save(image_path)

    # Celery 비동기 작업 실행
    process_receipt_task.delay(session)

    return jsonify({"session": session})


@app.route("/process", methods=["GET"])
def process_status():
    session = request.args.get("session")
    if not session:
        return jsonify({"error": "Missing session"}), 400

    result_path = os.path.join(BASE_DIR, session, "result.json")
    if not os.path.exists(result_path):
        return jsonify({"status": "processing"})

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)


# pip install paddlepaddle paddleocr
# pip install gunicorn
# gunicorn -w 4 -b 0.0.0.0:8800 --timeout 120 app:app