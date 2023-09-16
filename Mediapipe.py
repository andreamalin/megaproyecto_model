import mediapipe as mp
import matplotlib as plt
import numpy as np
import subprocess
import cv2
import glob, os
import pandas as pd

class MediapipeHands():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
    )
    hands = HandLandmarker.create_from_options(options)
    two_hands_words = ["familia", "por favor", "ayuda", "amor", "casa", "escuela", "salud", "feliz"]

    def __init__(self, past_data_path=None) -> None:
        self.frames = []
        self.sequence_id = 0
        self.validation_sequence_ids = []
        self.initial_dir = os.getcwd()
        self.ids_without_required_hands = []

        if (past_data_path != None):
            if (os.path.exists(past_data_path)):
                past_df = pd.read_csv(past_data_path)
                self.sequence_id = past_df["sequence_id"].max()

    def extract_coordinates_from_dir(self, dir, is_val=False):
        os.chdir(dir)

        output_fps_path = f'{dir}/adjusted_fps_video.mp4'

        files_to_extract = glob.glob("*.mp4")
        print(f'Hay {len(files_to_extract)} videos a extraer coordenadas')
        for file in files_to_extract:
            if ("adjusted_fps_video.mp4" in file):
                return
                
            input_path = f'{dir}/{file}'
            self.sequence_id += 1
            if (is_val):
                self.validation_sequence_ids.append(self.sequence_id)

            if ("(" not in file):
                name = file.split(".")[0]
            else:
                name = file[0:file.index("(")].strip()


            self.change_to_30_fps(video_input_path=input_path, video_output_path=output_fps_path)
            self.extract_video(output_fps_path, name, self.sequence_id, input_path)

        os.chdir(self.initial_dir)

    def extract_coordinates_from_path(self, path):
        output_fps_path = f'{os.getcwd()}/adjusted_fps_video.mp4'
        input_path = path
        self.sequence_id += 1

        name = path.split(".")[0]

        self.change_to_30_fps(video_input_path=input_path, video_output_path=output_fps_path)
        self.extract_video(output_fps_path, name, self.sequence_id, input_path)

    def normalize_coordinates(self, coordinates, target, image_height):
        matplot_coordinates = []
        normalized_coordinates = []
        for x, y in coordinates:
            y = image_height - y
            matplot_coordinates.append([x, y])

        # Find the minimum and maximum values among the coordinates
        min_x, min_y = np.min(matplot_coordinates, axis=0)
        max_x, max_y = np.max(matplot_coordinates, axis=0)

        # Normalize the coordinates
        normalized_coordinates = (matplot_coordinates - np.array([min_x, min_y])) / np.array([max_x - min_x, max_y - min_y])
        
        # Convert coordinates for plotting
        # visualize_data(normalized_coordinates, target)
        return normalized_coordinates

    def visualize_data(self, normalized_coordinates, target):
        # Unzip normalized coordinates for plotting
        normalized_x, normalized_y = zip(*normalized_coordinates)
        
        # Plot the normalized coordinates
        plt.figure(figsize=(8, 6))
        plt.scatter(normalized_x, normalized_y, color='blue', label=f'{target}')
        plt.xlabel('Normalized X')
        plt.xlabel('Normalized X')
        plt.ylabel('Normalized Y')
        plt.title('Normalized Coordinates Plot')
        plt.legend()
        plt.grid(True)
        plt.show()

    def extract_video(self, video, target, sequence_id, real_path):
        added_rows = 0
        detected_two_hands = False
        # For webcam input:
        cap = cv2.VideoCapture(video)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            

            name = f'{os.getcwd()}/test_frame.png'
            cv2.imwrite(name, frame)
            mp_image = mp.Image.create_from_file(name)
            hand_landmarker_result = self.hands.detect(mp_image)

            
            if len(hand_landmarker_result.handedness) > 0:
                row_data = {
                    "sequence_id": sequence_id,
                    "target": target,
                    "file": real_path
                }
                hand_sides = ["Left", "Right"]
                for idx, landmarks in enumerate(hand_landmarker_result.hand_landmarks):
                    detected_pixels = []
                    hand_side = hand_sides[idx]
                    # Iterate through detected hand landmarks
                    for landmark_idx, landmark in enumerate(landmarks):
                        x, y = landmark.x, landmark.y
                        detected_pixels.append([x  * frame.shape[1], y * frame.shape[0]])
                        # Draw circles on the frame
                        cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)

                        
                    detected_pixels = self.normalize_coordinates(detected_pixels, target, frame.shape[0])
                    for i in range(len(detected_pixels)):
                        x, y = detected_pixels[i]
                        row_data[f'x_{hand_side}_hand_{i}'] =  x
                        row_data[f'y_{hand_side}_hand_{i}'] =  y
                    
                if (len(hand_landmarker_result.handedness) == 1):
                    for i in range(21):
                        x, y = [0, 0]
                        row_data[f'x_{hand_sides[1]}_hand_{i}'] =  x
                        row_data[f'y_{hand_sides[1]}_hand_{i}'] =  y
                

                added_rows += 1
                self.frames.append(row_data)
            # cv2.imshow('Hand Tracking', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            if (added_rows == 30):
                break
            
            if (len(hand_landmarker_result.handedness) == 2):
                detected_two_hands = True
        


        if (added_rows == 0):
            print("!! No hand detected in ", real_path)
        else:
            while (added_rows < 30):
                x, y = [0, 0]
                for i in range(21):
                    row_data[f'x_{hand_sides[0]}_hand_{i}'] =  x
                    row_data[f'y_{hand_sides[0]}_hand_{i}'] =  y
                    row_data[f'x_{hand_sides[1]}_hand_{i}'] =  x
                    row_data[f'y_{hand_sides[1]}_hand_{i}'] =  y
                self.frames.append(row_data)
                added_rows += 1
        
        if (not detected_two_hands and target in self.two_hands_words):
            print(">> No se detectaron las dos manos necesarias en ", real_path, " con id ", sequence_id)
            self.ids_without_required_hands.append(sequence_id)
            
        cap.release()
        cv2.destroyAllWindows()
    

    def get_length(self, filename):
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration", "-of",
                                "default=noprint_wrappers=1:nokey=1", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        return float(result.stdout)

    def change_to_30_fps(self, video_input_path, video_output_path):
        duration = 1/self.get_length(video_input_path)

        c = f'ffmpeg -loglevel 0 -y -itsscale {duration} -i "' + video_input_path + f'" -filter:v fps=fps=30 "' + video_output_path + '"'
        subprocess.call(c, shell=True)

    def get_padded_data(self):
        df = pd.DataFrame(self.frames)
        df['sequence_id'] = df['sequence_id'].astype(int)
        df = df[~df.sequence_id.isin(self.ids_without_required_hands)]
        return df