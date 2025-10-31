import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import os
import csv

def run_yolo_inference(weights_path, images_folder, output_folder=None, 
                      save_crops=False, save_labels=False, save_annotated=False,
                      conf_threshold=0.25, device='cpu'):
    """
    Run YOLO inference on all images in a folder with various output options
    and generate a CSV file with detections.
    """
    
    # Load YOLO model
    print(f"Loading YOLO model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Setup paths
    images_path = Path(images_folder)
    if output_folder is None:
        output_folder = images_path.parent / "inference_results"
    else:
        output_folder = Path(output_folder)
    
    # Create output directories
    output_folder.mkdir(exist_ok=True)
    crops_dir = output_folder / "crops" if save_crops else None
    labels_dir = output_folder / "labels" if save_labels else None
    annotated_dir = output_folder / "annotated" if save_annotated else None
    
    if save_crops:
        crops_dir.mkdir(exist_ok=True)
    if save_labels:
        labels_dir.mkdir(exist_ok=True)
    if save_annotated:
        annotated_dir.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Récupération de toutes les images (minuscules/majuscules gérées)
    image_files = [f for f in images_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {images_folder}")
        return {"total_images": 0, "total_detections": 0}
    
    print(f"Found {len(image_files)} images to process")
    
    total_detections = 0
    processed_images = 0
    
    # CSV setup
    csv_path = output_folder / "detections.csv"
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["dossier", "image", "detection_number", "score"])
        
        # Process each image
        for img_path in image_files:
            try:
                print(f"Processing: {img_path.name}")
                
                results = model(str(img_path), conf=conf_threshold, device=device)
                result = results[0]
                img = cv2.imread(str(img_path))
                img_height, img_width = img.shape[:2]
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    total_detections += len(boxes)
                    
                    for i, box in enumerate(boxes, start=1):
                        conf = box.conf.item()
                        detection_name = f"detection{i}"
                        writer.writerow([img_path.parent.name, img_path.name, detection_name, conf])
                    
                    # Save labels in YOLO format
                    if save_labels:
                        label_file = labels_dir / f"{img_path.stem}.txt"
                        with open(label_file, 'w') as f:
                            for box in boxes:
                                cls = int(box.cls.item())
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                center_x = ((x1 + x2) / 2) / img_width
                                center_y = ((y1 + y2) / 2) / img_height
                                width = (x2 - x1) / img_width
                                height = (y2 - y1) / img_height
                                f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")
                    
                    # Save cropped detections
                    if save_crops:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cropped = img[y1:y2, x1:x2]
                            crop_filename = f"{img_path.stem}_crop_{i:03d}.jpg"
                            crop_path = crops_dir / crop_filename
                            cv2.imwrite(str(crop_path), cropped)
                    
                    # Save annotated images
                    if save_annotated:
                        annotated_img = img.copy()
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                            cls = int(box.cls.item())
                            conf = box.conf.item()
                            class_name = model.names[cls] if model.names else str(cls)
                            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name}: {conf:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), 
                                          (x1 + label_size[0], y1), (0, 255, 0), -1)
                            cv2.putText(annotated_img, label, (x1, y1 - 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        annotated_path = annotated_dir / f"{img_path.stem}_annotated{img_path.suffix}"
                        cv2.imwrite(str(annotated_path), annotated_img)
                else:
                    # No detection → write a line with no-detection
                    writer.writerow([img_path.parent.name, img_path.name, "no-detection", ""])
                    
                    # Still create empty label file if saving labels
                    if save_labels:
                        label_file = labels_dir / f"{img_path.stem}.txt"
                        label_file.touch()
                
                processed_images += 1

            except Exception as e:
                print(f"Error processing {img_path.name}: {str(e)}")
                continue

    print(f"\nInference completed!")
    print(f"Processed images: {processed_images}/{len(image_files)}")
    print(f"Total detections: {total_detections}")
    print(f"CSV saved to: {csv_path}")
    print(f"Output folder: {output_folder}")
    
    return {
        "total_images": len(image_files),
        "processed_images": processed_images,
        "total_detections": total_detections,
        "output_folder": str(output_folder),
        "csv_path": str(csv_path)
    }

def main():
    parser = argparse.ArgumentParser(description="Run YOLO inference on a folder of images")
    parser.add_argument("weights", help="Path to YOLO weights file (.pt)")
    parser.add_argument("images", help="Path to folder containing images")
    parser.add_argument("-o", "--output", help="Output folder path (default: inference_results)")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped detection images")
    parser.add_argument("--save-labels", action="store_true", help="Save detection labels in YOLO format")
    parser.add_argument("--save-annotated", action="store_true", help="Save images with bounding boxes drawn")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (default: cpu)")
    
    args = parser.parse_args()
    
    results = run_yolo_inference(
        weights_path=args.weights,
        images_folder=args.images,
        output_folder=args.output,
        save_crops=args.save_crops,
        save_labels=args.save_labels,
        save_annotated=args.save_annotated,
        conf_threshold=args.conf,
        device=args.device
    )
    
    return results

if __name__ == "__main__":
    main()
