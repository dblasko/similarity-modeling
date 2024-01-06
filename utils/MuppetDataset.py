import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm


class MuppetDataset:
    """
    Wrapper class that contains the information of annotated Muppet Videos.
    """

    def __init__(
        self,
        video_paths,
        annotations_paths,
        extract_audio=False,
        extract_frames=True,
        frame_rate=25,
    ):
        self.frame_rate = frame_rate
        self.video_paths = video_paths
        self.annotations_paths = annotations_paths
        self.annotations = self.__load_annotations()
        if extract_audio:
            self.__extract_audio()
        if extract_frames:
            self.__extract_frames()
        self.__load_frames()
        self.__load_audio()  # sets self.audio_paths and self.audios - list of dicts {"audio": librosa_loaded_audio, "sr": sampling_rate}

    def __load_annotations(self):
        annotations = pd.DataFrame()
        for annotation_path in self.annotations_paths:
            ann = pd.read_csv(annotation_path, sep=";")
            annotations = pd.concat([annotations, ann], ignore_index=True)
        # Replace video IDs by their index in the dataset object
        # In the "Video" column, replace the 211 value with 0
        annotations["Video"] = (
            annotations["Video"].replace(211, 0).replace(244, 1).replace(343, 2)
        )
        return annotations

    def __load_frames(self):
        """
        Loads the frames of the video.
        """
        paths = []
        for video_idx, video_path in enumerate(self.video_paths):
            frames_dir = os.path.join(
                os.path.join(os.path.dirname(video_path), "video"), str(video_idx)
            )
            paths.append(frames_dir)
        self.frame_paths = paths

        self.frames = {}
        for video_idx, frame_folder in enumerate(self.frame_paths):
            ordered_frames = []
            for frame in os.listdir(frame_folder):
                frame_idx = int(frame.split("_")[1].split(".")[0])
                # append to the list of frames
                ordered_frames.append((frame_idx, frame))
            ordered_frames.sort(key=lambda tup: tup[0])
            self.frames[video_idx] = ordered_frames

    def __extract_frames(self):
        for video_idx, video_path in enumerate(self.video_paths):
            # Create video subdirectory if it does not exist
            video_dir = os.path.join(os.path.dirname(video_path), "video")
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            video_dir = os.path.join(video_dir, str(video_idx))
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            # Extract frames of the video to the folder
            capture = cv2.VideoCapture(video_path)
            success, frame = capture.read()
            pbar = tqdm(total=capture.get(7))
            frame_nr = 0
            while success:  # extraction at 25 fps
                cv2.imwrite(f"{video_dir}/frame_{frame_nr}.png", frame)
                success, frame = capture.read()
                frame_nr += 1
                pbar.update(1)
            capture.release()
            # remove the last frame, which is a duplicate
            os.remove(f"{video_dir}/frame_{frame_nr - 1}.png")

    def __load_audio(self):
        paths = []
        for video_path in self.video_paths:
            audio_path = os.path.join(
                os.path.dirname(video_path),
                "audio",
                os.path.basename(video_path).replace(".avi", ".wav"),
            )
            paths.append(audio_path)
        self.audio_paths = paths

        self.audios = []
        for audio_path in self.audio_paths:
            audio, sr = librosa.load(
                audio_path, sr=None
            )  # sr=None loads the original sampling rate
            self.audios.append({"audio": audio, "sr": sr})

    def __extract_audio(self):
        """
        Extracts the audio from the video files and saves it in an "audio" subdirectory.
        """
        for video_path in self.video_paths:
            # Create audio subdirectory if it does not exist
            audio_dir = os.path.join(os.path.dirname(video_path), "audio")
            if not os.path.exists(audio_dir):
                os.makedirs(audio_dir)
            audio_path = os.path.join(
                os.path.dirname(video_path),
                "audio",
                os.path.basename(video_path).replace(".avi", ".wav"),
            )
            try:
                os.system(
                    f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
                )  # f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path}"
            except Exception as e:
                print(e)
                print(
                    f"Could not extract audio from {video_path} - please make sure you have the ffmpeg executable installed and in your path."
                )
