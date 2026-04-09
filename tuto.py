import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import colors

from collections import defaultdict

class ObjectTracking:
    """Object Tracking using Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/"""

    def __init__(self, model_path="yolo26n.pt", source="path/to/video.mp4"):

        self.model = YOLO(model_path)  # Model initialization
        self.names = self.model.names  # Class names

        # Video capturing module
        self.cap = cv2.VideoCapture(source)
        assert self.cap.isOpened(), "Error reading video file"

        # Video writing module
        w, h, fps = (
            int(self.cap.get(x)) 
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH, 
                cv2.CAP_PROP_FRAME_HEIGHT, 
                cv2.CAP_PROP_FPS))
        self.writer = cv2.VideoWriter(
            "object-tracking.avi",
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, 
            (w, h)
            )

        self.track_history = defaultdict(lambda: [])  # Store the track history

        # Display settings
        self.rect_width=2
        self.font = 0.5
        self.text_width=2
        self.padding = 12
        self.margin = 10
        self.circle_thickness=5
        self.polyline_thickness=2

        # Window setup
        self.window_name = "YOLO Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_bbox(self, im0, box, track_id, cls):
        """Draw bounding box with label at TOP-LEFT, but TEXT CENTERED in its box."""

        x1, y1, x2, y2 = map(int, box)

        color = colors(int(cls), True)

        # Draw main bounding box
        cv2.rectangle(im0, (x1, y1), (x2, y2), color, self.rect_width)

        # Prepare label
        label = f"{self.names[int(cls)]}:{int(track_id)}"

        # Get text size
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font, self.text_width
        )

        bg_x1 = x1  # left edge of bbox
        bg_x2 = bg_x1 + (tw + 2 * self.padding)

        bg_y2 = y1  # top of bbox
        bg_y1 = bg_y2 - (th + 2 * self.margin)

        # Draw filled background rectangle (top-left)
        cv2.rectangle(
            im0,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )

        text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
        text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2  # small vertical tweak

        cv2.putText(
            im0,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.font,
            (104, 31, 17) if cls==2 else (255, 255, 255) ,  # white text
            self.text_width,
            cv2.LINE_AA,
        )

    def run(self):
        """Function to run object tracking on video file or webcam."""

        while self.cap.isOpened():
            success, im0 = self.cap.read()

            if not success:
                print("End of video or failed to read image.")
                break

            # --- NOUVEAUX PARAMÈTRES DE TRACKING (SPRINT 2) ---
            results = self.model.track(
                im0, 
                persist=True,
                tracker="bytetrack.yaml", # ByteTrack gère mieux les occlusions (quand le ballon passe derrière un joueur)
                conf=0.10,                # On abaisse le seuil de confiance à 10% pour forcer la détection du ballon
                iou=0.45,                 # Ajuste la façon dont les boîtes se chevauchent
                imgsz=960                 # CRUCIAL : on track avec la même résolution qu'à l'entraînement !
            )

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu()
                    ids = result.boxes.id.cpu()
                    clss = result.boxes.cls.tolist()

                    if boxes is not None or ids is not None:
                        for box, id, cls in zip(boxes, ids.tolist(), clss):
                            self.draw_bbox(im0, box, id, cls)

                            x1, y1, x2, y2 = box
                            track = self.track_history[id]

                            # append box centroid
                            track.append(
                                (float((x1+x2)/2), 
                                float((y1+y2)/2))
                                )  
                            if len(track) > 50:  # retain 30 tracks for 30 frames
                                track.pop(0)
    
                            # draw the tracking lines
                            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                            
                            cv2.circle(
                                im0,
                                (int(track[-1][0]), int(track[-1][1])),
                                5,
                                colors(cls, True),
                                -1
                            )

                            cv2.polylines(
                                im0, 
                                [points], 
                                isClosed=False, 
                                color=colors(cls, True), 
                                thickness=self.polyline_thickness
                                )

            self.writer.write(im0)
            cv2.imshow(self.window_name, im0)  # Display and handle input

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.selected_id = None
                print("Selection cleared")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()


# --- ETAPE 1 : ENTRAINEMENT (À faire une seule fois) ---
# --- ETAPE 1 : ENTRAINEMENT  ---
def train_model():
    model = YOLO("yolo26s.pt") 
    model.train(
        data='data/data.yaml', 
        epochs=100, 
        imgsz=960, 
        batch=4,          # <--- MODIFICATION : on passe de 16 à 4 (voire 2 si ça plante encore)
        workers=2,        # <--- NOUVEAU : limite le nombre de cœurs CPU qui chargent les données
        device='cpu',     
        amp=True,
        patience=20,
        mosaic=1.0,
        scale=0.5,
        hsv_v=0.4
    )

# --- ETAPE 2 : TRACKING ---
if __name__ == "__main__":
    # Décommente la ligne suivante si tu n'as pas encore entraîné ton modèle :
    train_model() 
    
    # "runs/detect/train/weights/best.pt" est le chemin par défaut après train
    tracker = ObjectTracking(model_path="runs/detect/train3/weights/best.pt", source="videos/video_basket.mp4")
    tracker.run()