import os
import cv2
import numpy as np
import time
import subprocess
import imageio_ffmpeg
from flask import Flask, render_template, Response, request, redirect, url_for, jsonify
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB limit

# Load custom YOLOv8 model (bestV2.pt)
model = YOLO('bestV2.pt')

VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask', 'no_helmet', 'no_vest']

os.makedirs('static', exist_ok=True)
os.makedirs('static/violations', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

last_saved_time = 0
violation_streak = 0
STREAK_THRESHOLD = 5

def check_and_save_violation_crop(frame, results):
    global last_saved_time, violation_streak
    current_time = time.time()
    
    violation_found = False
    target_box = None

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in VIOLATION_CLASSES or 'NO-' in class_name.upper():
                violation_found = True
                target_box = box
                break
        if violation_found:
            break

    if violation_found:
        violation_streak += 1
    else:
        violation_streak = 0

    if violation_streak >= STREAK_THRESHOLD and target_box is not None:
        if current_time - last_saved_time > 1.5:
            x1, y1, x2, y2 = map(int, target_box.xyxy[0])
            box_w = x2 - x1
            box_h = y2 - y1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            crop_size = int(max(box_w, box_h) * 1.8)
            h, w, _ = frame.shape
            crop_x1 = max(0, center_x - crop_size // 2)
            crop_y1 = max(0, center_y - crop_size // 2)
            crop_x2 = min(w, center_x + crop_size // 2)
            crop_y2 = min(h, center_y + crop_size // 2)
            
            cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if cropped.size > 0:
                cropped_360p = cv2.resize(cropped, (400, 400), interpolation=cv2.INTER_AREA)
                filename = f"violation_{int(current_time)}.jpg"
                filepath = os.path.join('static/violations', filename)
                cv2.imwrite(filepath, cropped_360p)
                
                last_saved_time = current_time
                violation_streak = 0
                print(f"[ALERT] Saved high-accuracy violation crop: {filepath}")

@app.route('/')
def home():
    return render_template('index.html', result_image=None, result_video=None)

@app.route('/get_violations')
def get_violations():
    folder = 'static/violations'
    files = sorted(os.listdir(folder), reverse=True) if os.path.exists(folder) else []
    images = [f'/static/violations/{f}' for f in files if f.endswith('.jpg')]
    return jsonify(images[:12])

@app.route('/upload_media', methods=['GET', 'POST'])
def upload_media():
    if request.method == 'GET':
        return redirect(url_for('home'))

    result_image = None
    result_video = None

    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is not None:
            results = model(frame, imgsz=640, verbose=False)
            check_and_save_violation_crop(frame, results) # Log violations from uploaded image
            res_plotted = results[0].plot()

            output_filename = 'scanned_' + file.filename
            output_path = os.path.join('static', output_filename)
            cv2.imwrite(output_path, res_plotted)
            result_image = '/' + output_path.replace('\\', '/')

    elif 'video' in request.files and request.files['video'].filename != '':
        file = request.files['video']
        input_path = os.path.join('uploads', file.filename)
        file.save(input_path)
        print(f"\n[INFO] Starting universal video processing for: {file.filename}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("[ERROR] Could not open uploaded video format.")
            return render_template('index.html', result_image=None, result_video=None)

        width, height, fps = 854, 480, 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        base_name = os.path.splitext(file.filename)[0]
        temp_output_path = os.path.join('static', 'temp_' + base_name + '.avi')
        final_output_filename = 'processed_' + base_name + '.mp4'
        final_output_path = os.path.join('static', final_output_filename)
        
        # Step 1: Write raw video via OpenCV MJPG
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            frame_resized = cv2.resize(frame, (width, height))
            results = model(frame_resized, imgsz=320, verbose=False)
            
            # Check and save violations during batch video scanning
            check_and_save_violation_crop(frame_resized, results)
            
            out.write(results[0].plot())
            
            frame_count += 1
            if total_frames > 0 and frame_count % 30 == 0:
                print(f"[INFO] Processed {frame_count} / {total_frames} frames...")

        cap.release()
        out.release()

        # Step 2: Convert/Transcode to universal H.264 MP4 using imageio_ffmpeg.get_ffmpeg_exe()
        print("[INFO] Transcoding video to universal MP4 format via Python FFmpeg...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_cmd = [
            ffmpeg_exe, '-y', '-i', temp_output_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k', final_output_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"[SUCCESS] Universal video scan complete: {final_output_path}\n")
        except Exception as e:
            print(f"[WARNING] Transcoding failed: {e}")
            final_output_path = temp_output_path
            final_output_filename = os.path.basename(temp_output_path)

        # Cleanup intermediate temp file if conversion succeeded
        if os.path.exists(temp_output_path) and final_output_path != temp_output_path:
            os.remove(temp_output_path)

        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            result_video = f'/static/{final_output_filename}?t={os.path.getmtime(final_output_path)}'

    return render_template('index.html', result_image=result_image, result_video=result_video)

@app.route('/video_feed')
def video_feed():
    return Response(generate_laptop_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_laptop_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                continue
            
            results = model(frame, imgsz=416, verbose=False)
            check_and_save_violation_crop(frame, results)
            
            annotated_frame = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                continue
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass
    finally:
        if cap.isOpened():
            cap.release()

@app.route('/process_ipad_frame', methods=['POST'])
def process_ipad_frame():
    try:
        file = request.files['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if frame is not None:
            results = model(frame, imgsz=416, verbose=False)
            check_and_save_violation_crop(frame, results)
            
            annotated = results[0].plot()
            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception:
        pass
    
    return '', 204

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
