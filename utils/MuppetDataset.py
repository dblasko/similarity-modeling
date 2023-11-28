import os
import pandas as pd
import librosa


class MuppetDataset:
    """
    Wrapper class that contains the information of annotated Muppet Videos.
    """

    def __init__(
        self, video_paths, annotations_paths, extract_audio=False, extract_frames=True
    ):
        self.video_paths = video_paths
        self.annotations_paths = annotations_paths
        self.annotations = self.__load_annotations()
        if extract_audio:
            self.__extract_audio()
        if extract_frames:
            self.__extract_frames()
        self.frames = self.__load_frames()
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
        # TODO:
        return None

    def __extract_frames(self):
        # TODO:
        pass

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
