import os
import uuid
from datetime import datetime
from urllib.request import urlretrieve

# Modules for detection
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist

# Modules for plotting graph
import csv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Modules for webserver
from flask import Flask, request, render_template, jsonify, url_for, send_file
from werkzeug.utils import secure_filename
import threading
from base64 import b64decode

###### CODE BEGINS ########

# Default config values
DEFAULT_MIN_DISTANCE = 50
DEFAULT_MIN_CONF = 0.3
DEFAULT_NMS_THRESH = 0.3

# GLOBAL status dict
processing_status = {}

# Initialize FLask
app = Flask(__name__)
app.config["SECRET_KEY"] = "CATZ420"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500mb
app.config["UPLOAD_FOLDER"] = "temp_uploads"
app.config["RESULTS_FOLDER"] = "results"

print("[INFO] Human Distance Monitoring System Server...")
print(
    f"[INFO] Default: MIN_DISTANCE={DEFAULT_MIN_DISTANCE}, MIN_CONF={DEFAULT_MIN_CONF}, NMS_THRESH={DEFAULT_NMS_THRESH}"
)

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)


def cleanup_folders():
    print("[INFO] Cleaning upload and results folders...")

    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to delete {filename}: {e}")
    for filename in os.listdir(app.config["RESULTS_FOLDER"]):
        file_path = os.path.join(app.config["RESULTS_FOLDER"], filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to delete {filename}: {e}")
    return True


# Allowed files to upload
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {
        "mp4",
        "avi",
        "mov",
        "mkv",
        "webm",
    }


##### CODE/CORE START #####

# Setup YOLO
def setup_yolo():
    # Base Directory

    base_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(base_dir, "yolo-coco")
    os.makedirs(yolo_dir, exist_ok=True)

    cocoPath = os.path.join(yolo_dir, "coco.names")
    coco_cfg = os.path.join(yolo_dir, "yolov3.cfg")
    coco_weights = os.path.join(yolo_dir, "yolov3.weights")

    base_link = (
        "https://github.com/htrafsan/Human-Distance-Detector/releases/download/1"
    )

    # Download coco.names if not exists

    if not os.path.isfile(cocoPath):
        print("[-] coco.names not found. Downloading..")
        try:
            urlretrieve(f"{base_link}/coco.names", cocoPath)
        except Exception as e:
            print(f"[ERROR] Failed to download coco.names: {e}")
            return None, None, None
    LABELS = open(cocoPath).read().strip().split("\n")

    # Download yolov3.cfg if not exists

    if not os.path.isfile(coco_cfg):
        print("[-] yolov3.cfg not found. Downloading..")
        try:
            urlretrieve(f"{base_link}/yolov3.cfg", coco_cfg)
        except Exception as e:
            print(f"[ERROR] Failed to download yolov3.cfg: {e}")
            return None, None, None
    # Download yolov3.weights if not exists

    if not os.path.isfile(coco_weights):
        print("[-] yolov3.weights not found. Downloading..")
        try:
            urlretrieve(f"{base_link}/yolov3.weights", coco_weights)
        except Exception as e:
            print(f"[ERROR] Failed to download yolov3.weights: {e}")
            return None, None, None
    # Load YOLO Detector

    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(coco_cfg, coco_weights)
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln, LABELS


# Module for detecting people
def detect_people(frame, net, ln, personIdx=0, min_conf=0.3, nms_thresh=0.3):
    try:
        (H, W) = frame.shape[:2]
        results = []

        """
        1. created a blob from the input frame
		2. perform a forward pass of the YOLO object detector
		3. giving us bounding boxes & associated probabilities
		"""

        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False
        )
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        centroids = []
        confidences = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if classID == personIdx and confidence > min_conf:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        # Apply non-maxima suppression

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, min_conf, nms_thresh)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)
        return results
    except Exception as e:
        print(f"[ERROR] Error in detect_people: {e}")
        return []


# Module to process video
def process_video_file(
    input_path,
    output_video_path,
    output_csv_path,
    job_id,
    is_bw=False,
    min_distance=50,
    min_conf=0.3,
    nms_thresh=0.3,
):
    try:
        # Setup YOLO
        net, ln, LABELS = setup_yolo()

        if net is None:
            processing_status[job_id]["status"] = "error"
            processing_status[job_id]["error"] = "Failed to load YOLO"
            return False
        # Initialize video

        vs = cv2.VideoCapture(input_path)
        if not vs.isOpened():
            processing_status[job_id]["status"] = "error"
            processing_status[job_id][
                "error"
            ] = f"Could not open video file: {input_path}"
            return False
        writer = None
        frame_results = []
        frame_number = 0

        processing_status[job_id]["status"] = "processing"

        # Get total frames count
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        processing_status[job_id]["total_frames"] = total_frames

        while True:
            # If cancel api is called, it will break

            if processing_status[job_id].get("cancelled", False):
                break
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            if is_bw:
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = imutils.resize(frame, width=700)

            try:
                # Detect Person
                person_index = LABELS.index("person")
                results = detect_people(
                    frame,
                    net,
                    ln,
                    personIdx=person_index,
                    min_conf=min_conf,
                    nms_thresh=nms_thresh,
                )
            except ValueError:
                results = []
            violate = set()
            if len(results) >= 2:
                centroids = np.array([r[2] for r in results])
                D = dist.cdist(centroids, centroids, metric="euclidean")

                for i in range(0, D.shape[0]):
                    for j in range(i + 1, D.shape[1]):
                        if D[i, j] < min_distance:
                            violate.add(i)
                            violate.add(j)
            for i, (prob, bbox, centroid) in enumerate(results):
                (startX, startY, endX, endY) = bbox
                (cX, cY) = centroid
                color = (0, 255, 0) if i not in violate else (0, 0, 255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, color, 1)
            # Local variable to store

            num_persons = len(results)
            num_green = num_persons - len(violate)
            num_red = len(violate)
            frame_results.append([frame_number, num_persons, num_green, num_red])

            text_green = f"Safe: {num_green}"
            text_red = f"Violations: {num_red}"
            # Add safe, violations to video

            cv2.putText(
                frame,
                text_green,
                (10, frame.shape[0] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                3,
            )
            cv2.putText(
                frame,
                text_red,
                (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                3,
            )

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(
                    output_video_path,
                    fourcc,
                    vs.get(cv2.CAP_PROP_FPS) or 25,
                    (frame.shape[1], frame.shape[0]),
                    True,
                )
            if writer is not None:
                writer.write(frame)
            # Log process to api

            frame_number += 1
            progress = (
                int((frame_number / total_frames) * 100) if total_frames > 0 else 0
            )
            processing_status[job_id]["progress"] = progress
            processing_status[job_id]["current_frame"] = frame_number
        vs.release()
        if writer is not None:
            writer.release()
        # Push data to csv file

        if frame_results:
            with open(output_csv_path, "w", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(["frame", "total_persons", "safe", "violations"])
                writer_csv.writerows(frame_results)
        # Update api data

        processing_status[job_id]["status"] = "completed"
        processing_status[job_id]["output_video"] = output_video_path
        processing_status[job_id]["output_csv"] = output_csv_path
        processing_status[job_id]["total_frames_processed"] = frame_number

        return True
    except Exception as e:
        print(f"[ERROR] Error processing video: {e}")
        processing_status[job_id]["status"] = "error"
        processing_status[job_id]["error"] = str(e)
        return False


# Module to calculate metrics and plot chart
def calculate_metrics(train_csv, bw_csv):
    try:
        tp_total, tn_total, fp_total, fn_total = 0, 0, 0, 0
        frame_count = 0

        # Read from csv

        with open(train_csv, "r") as f_train, open(bw_csv, "r") as f_bw:
            train_reader = csv.DictReader(f_train)
            bw_reader = csv.DictReader(f_bw)

            for train_row, bw_row in zip(train_reader, bw_reader):
                try:
                    gt_safe = int(train_row["safe"])
                    gt_viol = int(train_row["violations"])
                    pred_safe = int(bw_row["safe"])
                    pred_viol = int(bw_row["violations"])

                    tp = min(gt_viol, pred_viol)
                    tn = min(gt_safe, pred_safe)
                    fp = max(0, pred_viol - gt_viol)
                    fn = max(0, gt_viol - pred_viol)

                    tp_total += tp
                    tn_total += tn
                    fp_total += fp
                    fn_total += fn
                    frame_count += 1
                except (ValueError, KeyError) as e:
                    continue
        # Equations

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0

        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        tnr = tn_total / (tn_total + fp_total) if (tn_total + fp_total) > 0 else 0

        accuracy = (
            (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total)
            if (tp_total + tn_total + fp_total + fn_total) > 0
            else 0
        )

        metrics = {
            "total_frames": frame_count,
            "tp_total": tp_total,
            "tn_total": tn_total,
            "fp_total": fp_total,
            "fn_total": fn_total,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "tnr": round(tnr, 4),
            "accuracy": round(accuracy, 4),
        }

        # Generate comparison plot in results folder

        plot_filename = f"comparison_plot_{uuid.uuid4().hex[:8]}.png"
        plot_path = os.path.join(app.config["RESULTS_FOLDER"], plot_filename)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        metric_names = ["Precision", "Recall", "F1 Score", "TNR", "Accuracy"]
        metric_values = [precision, recall, f1_score, tnr, accuracy]
        bars = plt.bar(
            metric_names,
            metric_values,
            color=["blue", "green", "red", "purple", "orange"],
        )
        plt.ylim(0, 1.05)
        plt.title("Detection Metrics")
        plt.ylabel("Score")
        plt.xticks(rotation=45)

        for bar, value in zip(bars, metric_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        plt.subplot(1, 2, 2)
        count_names = ["TP", "TN", "FP", "FN"]
        count_values = [tp_total, tn_total, fp_total, fn_total]
        bars = plt.bar(
            count_names,
            count_values,
            color=["lightblue", "lightgreen", "lightcoral", "lightsalmon"],
        )
        plt.title("Detection Counts")
        plt.ylabel("Count")

        for bar, value in zip(bars, count_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{value}",
                ha="center",
                va="bottom",
            )
        plt.suptitle("Color vs BW Video Detection Comparison")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        metrics["plot_path"] = plot_path
        return metrics
    except Exception as e:
        print(f"[ERROR] Error calculating metrics: {e}")
        return None


##### CODE/CORE END #####


##### CODE/FLASK START #####


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_min_distance=DEFAULT_MIN_DISTANCE,
        default_min_conf=DEFAULT_MIN_CONF,
        default_nms_thresh=DEFAULT_NMS_THRESH,
    )


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload files to the server"""
    try:
        if "color_file" not in request.files or "bw_file" not in request.files:
            return jsonify({"error": "Both colored and BW files are required"}), 400
        color_file = request.files["color_file"]
        bw_file = request.files["bw_file"]

        if color_file.filename == "" or bw_file.filename == "":
            return jsonify({"error": "Both files must be selected"}), 400
        if not (
            color_file
            and allowed_file(color_file.filename)
            and bw_file
            and allowed_file(bw_file.filename)
        ):
            return jsonify({"error": "Invalid file type"}), 400
        min_distance = request.form.get("min_distance", 50, type=float)
        min_conf = request.form.get("min_conf", 0.3, type=float)
        nms_thresh = request.form.get("nms_thresh", 0.3, type=float)

        job_id = str(uuid.uuid4())[:8]

        # Save files to upload folder

        color_filename = secure_filename(color_file.filename)
        color_file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"color_{job_id}_{color_filename}"
        )
        color_file.save(color_file_path)

        bw_filename = secure_filename(bw_file.filename)
        bw_file_path = os.path.join(
            app.config["UPLOAD_FOLDER"], f"bw_{job_id}_{bw_filename}"
        )
        bw_file.save(bw_file_path)

        # Create results in results folder
        color_output_video = os.path.join(
            app.config["RESULTS_FOLDER"], f"color_{job_id}.avi"
        )
        color_output_csv = os.path.join(
            app.config["RESULTS_FOLDER"], f"color_{job_id}.csv"
        )
        bw_output_video = os.path.join(app.config["RESULTS_FOLDER"], f"bw_{job_id}.avi")
        bw_output_csv = os.path.join(app.config["RESULTS_FOLDER"], f"bw_{job_id}.csv")

        processing_status[job_id] = {
            "status": "uploaded",
            "color_filename": color_filename,
            "bw_filename": bw_filename,
            "upload_time": datetime.now().isoformat(),
            "progress": 0,
            "current_frame": 0,
            "total_frames": 0,
            "current_task": "Processing started",
            "parameters": {
                "min_distance": min_distance,
                "min_conf": min_conf,
                "nms_thresh": nms_thresh,
            },
        }

        def process_in_background():
            try:
                processing_status[job_id]["current_task"] = "Processing colored video"
                color_success = process_video_file(
                    color_file_path,
                    color_output_video,
                    color_output_csv,
                    job_id,
                    is_bw=False,
                    min_distance=min_distance,
                    min_conf=min_conf,
                    nms_thresh=nms_thresh,
                )

                if color_success:
                    processing_status[job_id]["current_task"] = "Processing BW video"
                    processing_status[job_id]["progress"] = 0
                    bw_success = process_video_file(
                        bw_file_path,
                        bw_output_video,
                        bw_output_csv,
                        job_id,
                        is_bw=True,
                        min_distance=min_distance,
                        min_conf=min_conf,
                        nms_thresh=nms_thresh,
                    )

                    if bw_success:
                        processing_status[job_id][
                            "current_task"
                        ] = "Both videos processed"
                        metrics = calculate_metrics(color_output_csv, bw_output_csv)

                        if metrics:
                            processing_status[job_id]["status"] = "completed"
                            processing_status[job_id]["metrics"] = metrics
                            processing_status[job_id][
                                "color_output_video"
                            ] = color_output_video
                            processing_status[job_id][
                                "bw_output_video"
                            ] = bw_output_video
                            processing_status[job_id][
                                "color_output_csv"
                            ] = color_output_csv
                            processing_status[job_id]["bw_output_csv"] = bw_output_csv
                            processing_status[job_id]["comparison_plot"] = metrics[
                                "plot_path"
                            ]
                        else:
                            processing_status[job_id]["status"] = "error"
                            processing_status[job_id][
                                "error"
                            ] = "Failed to calculate metrics"
                    else:
                        processing_status[job_id]["status"] = "error"
                        processing_status[job_id][
                            "error"
                        ] = "Failed to process BW video"
                else:
                    processing_status[job_id]["status"] = "error"
                    processing_status[job_id][
                        "error"
                    ] = "Failed to process colored video"
            except Exception as e:
                processing_status[job_id]["status"] = "error"
                processing_status[job_id]["error"] = str(e)

        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()

        return jsonify(
            {
                "success": True,
                "message": "Files uploaded and processing started",
                "job_id": job_id,
                "redirect": url_for("results", job_id=job_id),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500


@app.route("/results/<job_id>")
def results(job_id):
    if job_id not in processing_status:
        return "Job not found.", 404
    return render_template("results.html", job_id=job_id)


@app.route("/status/<job_id>")
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({"error": "Job not found"}), 404
    status = processing_status[job_id].copy()
    if "metrics" in status:
        status["metrics"] = {
            k: v for k, v in status["metrics"].items() if k != "plot_path"
        }
    return jsonify(status)


@app.route("/cancel/<job_id>")
def cancel_job(job_id):
    if job_id in processing_status:
        processing_status[job_id]["cancelled"] = True
        processing_status[job_id]["status"] = "cancelled"
        return jsonify({"success": True, "message": "Job cancelled"})
    else:
        return jsonify({"error": "Job not found"}), 404


@app.route("/download/<path:encoded_path>")
def download_file(encoded_path):
    try:
        file_path = b64decode(encoded_path).decode("utf-8")
        if os.path.exists(file_path) and (
            file_path.startswith(app.config["UPLOAD_FOLDER"])
            or file_path.startswith(app.config["RESULTS_FOLDER"])
        ):
            return send_file(file_path, as_attachment=True)
        else:
            return "File not found", 404
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500


@app.route("/clean")
def clean_all():
    try:
        result = cleanup_folders()

        return jsonify(
            {"success": result, "message": "All folders cleaned successfully"}
        )
    except Exception as e:
        return (
            jsonify({"success": result, "error": f"Failed to clean folders: {str(e)}"}),
            500,
        )


if __name__ == "__main__":
    # cleanup_folders()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
