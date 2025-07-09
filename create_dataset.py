import os
import cv2
from deepface import DeepFace

def prepare_dataset(raw_images_dir, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    
    for person_dir in os.listdir(raw_images_dir):
        person_path = os.path.join(raw_images_dir, person_dir)
        if os.path.isdir(person_path):
            output_dir = os.path.join(dataset_dir, person_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                try:
                    # Detect and align face
                    face = DeepFace.extract_faces(
                        img_path=img_path,
                        detector_backend="retinaface",
                        align=True
                    )[0]
                    
                    # Save processed face
                    output_path = os.path.join(output_dir, img_file)
                    cv2.imwrite(output_path, face['face'])
                    print(f"Processed: {output_path}")
                except Exception as e:
                    print(f"Skipped {img_path}: {str(e)}")

if __name__ == "__main__":
    prepare_dataset("raw_images", "face_dataset")