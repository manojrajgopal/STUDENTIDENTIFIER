from flask import Flask, request, jsonify, send_from_directory
import os
import cv2
import numpy as np
import base64
import tempfile
from scipy.spatial.distance import cosine
from deepface import DeepFace
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'input_images'
OUTPUT_FOLDER = 'output'
DATASET_FOLDER = 'face_dataset'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Model configuration
MODELS = {
    "ArcFace": {"threshold": 0.4},
    "Facenet": {"threshold": 0.4},
    "Facenet512": {"threshold": 0.3},
    "VGG-Face": {"threshold": 0.5}
}

# Preload models
for model_name in MODELS:
    MODELS[model_name]["model"] = DeepFace.build_model(model_name)

# Helper functions
def verify_image(img):
    """Ensure image is in correct format"""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def extract_faces(image_path):
    """Improved face extraction with multiple detector fallbacks"""
    detectors = ["retinaface", "mtcnn", "opencv"]
    for detector in detectors:
        try:
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector,
                enforce_detection=False,
                align=True,
                expand_percentage=10
            )
            valid_faces = []
            for face in faces:
                if face['confidence'] > 0.9:
                    face_img = verify_image(face['face'])
                    valid_faces.append({
                        'face': face_img,
                        'area': face['facial_area']
                    })
            if valid_faces:
                print(f"Found {len(valid_faces)} faces using {detector}")
                return valid_faces
        except Exception as e:
            print(f"{detector} failed: {str(e)}")
    return []

def load_known_faces():
    """Load known faces with all models"""
    known = {}
    for model_name in MODELS:
        known[model_name] = []
        for file in os.listdir(DATASET_FOLDER):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(DATASET_FOLDER, file)
                try:
                    name = os.path.splitext(file)[0]
                    embedding = DeepFace.represent(
                        img_path=path,
                        model_name=model_name,
                        enforce_detection=False
                    )[0]["embedding"]
                    known[model_name].append((name, embedding))
                except Exception as e:
                    print(f"Error loading {file} for {model_name}: {str(e)}")
    return known

def match_faces(image_path):
    """Complete face matching pipeline"""
    known_faces = load_known_faces()
    faces = extract_faces(image_path)
    if not faces:
        return []
    
    results = []
    for i, face in enumerate(faces):
        face_id = f"face_{i+1}"
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{face_id}.jpg")
        cv2.imwrite(temp_path, face['face'])
        
        model_votes = {}
        for model_name, config in MODELS.items():
            try:
                # Get embedding for current face
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name=model_name,
                    enforce_detection=False
                )[0]["embedding"]
                
                # Find best match
                best_name = "Unknown"
                best_score = float('inf')
                for name, known_emb in known_faces[model_name]:
                    score = cosine(embedding, known_emb)
                    if score < best_score:
                        best_score = score
                        best_name = name
                
                confidence = 1 - best_score
                if confidence > config["threshold"]:
                    if best_name not in model_votes:
                        model_votes[best_name] = {"count": 0, "total_conf": 0}
                    model_votes[best_name]["count"] += 1
                    model_votes[best_name]["total_conf"] += confidence
                    
            except Exception as e:
                print(f"{model_name} failed for {face_id}: {str(e)}")
        
        os.remove(temp_path)
        
        # Determine final match
        if model_votes:
            best_match = max(
                model_votes.items(),
                key=lambda x: (x[1]["count"], x[1]["total_conf"])
            )[0]
            avg_conf = model_votes[best_match]["total_conf"] / model_votes[best_match]["count"]
        else:
            best_match = "Unknown"
            avg_conf = 0.0
        
        results.append({
            "id": face_id,
            "name": best_match,
            "confidence": avg_conf,
            "bbox": face['area']
        })
    
    return results

# Routes
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/capture', methods=['POST'])
def capture_image():
    try:
        data = request.json
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save image
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        image_path = os.path.join(UPLOAD_FOLDER, 'captured.jpg')
        cv2.imwrite(image_path, img)
        
        return jsonify({"status": "success", "path": image_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/identify', methods=['GET', 'POST'])
def identify_faces():
    try:
        image_path = os.path.join(UPLOAD_FOLDER, 'captured.jpg')
        if not os.path.exists(image_path):
            return jsonify({"error": "No image captured"}), 400
        
        results = match_faces(image_path)
        
        if not results:
            return jsonify({"status": "no_faces"})
        
        # Save annotated image
        img = cv2.imread(image_path)
        for face in results:
            bbox = face['bbox']
            color = (0, 255, 0) if face['name'] != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (bbox['x'], bbox['y']), 
                         (bbox['x']+bbox['w'], bbox['y']+bbox['h']), color, 2)
            cv2.putText(img, f"{face['name']} ({face['confidence']:.0%})",
                       (bbox['x'], bbox['y']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        output_path = os.path.join(OUTPUT_FOLDER, 'annotated.jpg')
        cv2.imwrite(output_path, img)
        
        return jsonify({
            "status": "success",
            "results": [(r['name'], r['bbox'], r['confidence']) for r in results]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/results')
def results():
    image_path = os.path.join(UPLOAD_FOLDER, 'captured.jpg')
    if not os.path.exists(image_path):
        return jsonify({"results": [], "done": True})
    
    return jsonify({
        "results": [(r['name'], r['bbox'], r['confidence']) for r in match_faces(image_path)],
        "done": True
    })

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, 'captured.jpg')
            file.save(image_path)
            
            # Verify image is readable
            img = cv2.imread(image_path)
            if img is None:
                os.remove(image_path)
                return jsonify({"error": "Invalid image file"}), 400
                
            return jsonify({
                "status": "success",
                "path": image_path,
                "filename": filename,
                "dimensions": f"{img.shape[1]}x{img.shape[0]}"
            })
        else:
            return jsonify({"error": "Allowed file types are jpg, jpeg, png"}), 400
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/process_upload', methods=['GET'])
def process_upload():
    try:
        image_path = os.path.join(UPLOAD_FOLDER, 'uploaded.jpg')
        if not os.path.exists(image_path):
            return jsonify({"error": "No uploaded image found"}), 404
            
        results = match_faces(image_path)
        if not results:
            return jsonify({"status": "no_faces"})
            
        # Save annotated image
        img = cv2.imread(image_path)
        for face in results:
            bbox = face['bbox']
            color = (0, 255, 0) if face['name'] != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (bbox['x'], bbox['y']), 
                         (bbox['x']+bbox['w'], bbox['y']+bbox['h']), color, 2)
            cv2.putText(img, f"{face['name']} ({face['confidence']:.0%})",
                       (bbox['x'], bbox['y']-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        output_path = os.path.join(OUTPUT_FOLDER, 'annotated.jpg')
        cv2.imwrite(output_path, img)
        
        return jsonify({
            "status": "success",
            "results": [(r['name'], r['bbox'], r['confidence']) for r in results]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register', methods=['POST'])
def register_face():
    try:
        # Get data from request
        data = request.json
        name = data.get('name', '').strip()
        image_data = data.get('image', '')
        
        # Validate inputs
        if not name:
            return jsonify({"error": "Name is required"}), 400
            
        if not image_data or not image_data.startswith('data:image'):
            return jsonify({"error": "Invalid image data"}), 400

        # Process image data
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        
        # Convert to numpy array and ensure proper format
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to 8-bit unsigned if needed
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        # Create filename and save
        filename = f"{name.replace(' ', '_')}.jpg"
        save_path = os.path.join(DATASET_FOLDER, filename)
        
        # Ensure directory exists
        os.makedirs(DATASET_FOLDER, exist_ok=True)
        
        # Save the image
        if not cv2.imwrite(save_path, img):
            return jsonify({"error": "Failed to save image"}), 500
            
        return jsonify({
            "status": "success",
            "message": "Person registered successfully",
            "filename": filename
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)