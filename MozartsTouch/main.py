import io
from pathlib import Path
import datetime
from PIL import Image
import time
import argparse
import yaml
from loguru import logger
from moviepy import VideoFileClip, AudioFileClip
import os
import logging
import json
import pandas as pd


'''
Because of Python's feature of chain importing (https://stackoverflow.com/questions/5226893/understanding-a-chain-of-imports-in-python)
you need to use these lines below instead of those above to be able to run the test code after `if __name__=="__main__"`
'''
if __name__=="__main__":
    from utils.image_processing import ImageRecognization
    from utils.music_generation import MusicGenerator,  MusicGeneratorFactory
    from utils.txt_converter import TxtConverter
    from utils.preprocess_single import PreProcessVideos
else: 
    from .utils.image_processing import ImageRecognization
    from .utils.music_generation import MusicGenerator,  MusicGeneratorFactory
    from .utils.txt_converter import TxtConverter
    from .utils.preprocess_single import PreProcessVideos


module_path = Path(__file__).resolve().parent 
with open(module_path / 'config.yaml', 'r', encoding='utf8') as file:
    config = yaml.safe_load(file)

test_mode =  config.get('TEST_MODE', False)
logger.info(f"Test mode: {test_mode}")

def import_ir():
    # '''导入图像识别模型'''
    # Import the image recognition model
    ir = ImageRecognization(test_mode=test_mode)
    return ir

def import_music_generator():
    start_time = time.time()
    music_model = config['DEFAULT_MUSIC_MODEL']
    if test_mode:
        mg = MusicGeneratorFactory.create_music_generator("test")
    else:
        mg = MusicGeneratorFactory.create_music_generator(music_model)
    logger.info(f"[TIME] taken to load Music Generation module {music_model}: {time.time() - start_time :.2f}s")
    return mg


class Entry:
    # '''每个Entry代表一次用户输入，然后调用自己的方法对输入进行处理以得到生成结果'''
    """
    Each Entry represents a user input, which is then processed by its own method to generate the result.
    """
    def __init__(self, image_recog:ImageRecognization, music_gen: MusicGenerator,\
                  music_duration: int, addtxt:str, output_folder:Path, img:Image=None, video_path:Path=None) -> None:
        self.img = img
        self.video_path = video_path
        self.txt = None
        self.txt_con = TxtConverter()
        self.converted_txt = None
        self.addtxt = addtxt  # Append text input # 追加文本输入
        self.image_recog = image_recog  # Use the provided image recognition model object # 使用传入的图像识别模型对象
        self.music_gen = music_gen  # Use the provided music generation object # 使用传入的音乐生成对象
        self.music_duration = music_duration
        self.output_folder = output_folder
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Record the user's upload time as an identifier # 记录用户上传时间作为标识符
        self.result_urls = None
        self.music_bytes_io = None

    def init_video(self):
        assert self.img is None
        # 将视频帧进行采样并分别识别，同时获取视频长度作为self.music_duration，之后调用self.video_txt_descriper合成描述视频的一段话
        # Sample the video frames and recognize them individually, while also obtaining the video length as self.music_duration.
        # Then, call self.video_txt_descripter to generate a description of the video.
        video_processor = PreProcessVideos(
            str(self.video_path), 
            self.image_recog, 
            prompt_amount=config['CAPTION_MODEL_CONFIG']['VIDEO_SAMPLE_AMOUNT']
        )
        video_frame_texts = video_processor.process_video()
        self.music_duration = video_processor.video_seconds + 1
        self.video_txt_descriper(video_frame_texts)
    
    def init_txt(self, caption):
        self.txt = caption


    def img2txt(self):
        assert self.video_path is None
        # '''进行图像识别'''
        # Perform image recognition
        self.txt = self.image_recog.img2txt(self.img)
    
    def txt_converter(self):
        # '''利用LLM优化已有的图片描述文本'''
        # Use LLM to optimize the existing image description text
        self.converted_txt = self.txt_con.txt_converter(self.txt, self.addtxt) # Append an additional input, specific changes can be found in txt_converter # 追加一个附加输入，具体改动参见txt_converter

    def video_txt_descriper(self, texts):
        # '''将每帧的描述文本转换为视频整体的描述文本'''
        # Convert the description text of each frame into an overall description of the video
        self.txt = self.txt_con.process_video_description(texts)
        logger.info(f"Video description: {self.txt}")

    def video_txt_converter(self):
        # '''利用LLM优化已有的视频描述文本'''
        # Use LLM to optimize the existing video description text
        self.converted_txt = self.txt_con.video_txt_converter(self.txt, self.addtxt) # 追加一个附加输入，具体改动参见txt_converter

    def txt2music(self):
        # '''根据文本进行音乐生成，获取生成的音乐的BytesIO或URL'''
        # Generate music based on the text and obtain the generated music's BytesIO or URL
        assert self.music_duration
        if self.music_gen.model_name.startswith("Suno"):
            self.result_urls = self.music_gen.generate(self.converted_txt, self.music_duration)
        else:
            self.music_bytes_io = self.music_gen.generate(self.converted_txt, self.music_duration)

    def save_to_file(self, file_name=None):
        # '''将音乐保存到`/outputs`中，文件名为用户上传时间的时间戳'''
        # Save the music to `/outputs` with the filename as the timestamp of the user's upload time
        self.output_folder.mkdir(parents=True, exist_ok=True)

        if file_name:
            self.result_file_name = file_name
        else:
            self.result_file_name = f"{self.timestamp}.wav"
        file_path = self.output_folder / self.result_file_name

        with open(file_path, "wb") as music_file:
            music_file.write(self.music_bytes_io.getvalue())

        # logger.info(f"音乐已保存至 {file_path}")
        logger.info(f"Music has been saved to {file_path}")

        return self.result_file_name
    def merge_audio_video(self):
        # '''合成原视频与生成的音乐'''
        # 读取视频文件
        # Combine the original video with the generated music
        # Read the video file
        video_clip = VideoFileClip(self.video_path)
        
        # 读取音频数据流
        # Read the audio data stream
        audio_clip = AudioFileClip(self.output_folder / self.result_file_name)
        
        # 将音频和视频合成
        # Combine the audio and video
        final_video = video_clip.with_audio(audio_clip)
        self.result_video_name = f"{self.timestamp}.mp4"
        # 输出到文件
        # Output to file
        final_video.write_videofile(video_file_path := self.output_folder / self.result_video_name)
        # logger.info(f"视频已保存至 {video_file_path}")
        logger.info(f"Video has been saved to {video_file_path}")
        return self.result_video_name

def img_to_music_generate(img: Image, music_duration: int, image_recog: ImageRecognization,\
                           music_gen: MusicGenerator, output_folder=Path("./outputs"), addtxt: str=None, file_name=None):
    # '''模型核心过程'''
    # # 根据输入mode信息获得对应的音乐生成模型类的实例
    # # mg = mgs[mode]

    # # 根据用户输入创建一个类，并传入图像识别和音乐生成模型的实例
    """
    Model core process
    """
    # Get the corresponding music generation model class instance based on the input mode information
    # mg = mgs[mode]

    # Create a class based on user input and pass the instances of the image recognition and music generation models

    entry = Entry(image_recog, music_gen, music_duration, addtxt, output_folder, img=img)

    # 图片转文字
    # Image to text
    entry.img2txt()

    # 文本优化
    # Text optimization
    entry.txt_converter()

    #文本生成音乐
    # Text to music generation
    entry.txt2music()

    if not music_gen.model_name.startswith("Suno"):
        # print("Here.")
        entry.save_to_file(file_name=file_name)

    # return (entry.txt, entry.converted_txt, entry.result_file_name)
    return (None, None, None)

def video_to_music_generate(video_path: Path, image_recog: ImageRecognization, music_gen: MusicGenerator,\
                             output_folder=Path("./outputs"), addtxt: str=None, file_name=None):
    # '''模型核心过程'''
    # # 根据用户输入创建一个类，并传入图像识别和音乐生成模型的实例
    """
    Model core process
    """
    # Create a class based on user input and pass the instances of the image recognition and music generation models
    entry = Entry(image_recog, music_gen, None, addtxt, output_folder, video_path=video_path)
    # 视频采样、识别
    # Video sampling and recognition
    entry.init_video()

    # 文本优化
    # Text optimization
    entry.video_txt_converter()

    # 文本生成音乐
    # Text to music generation
    entry.txt2music()
    entry.save_to_file(file_name=file_name)

    # 合成视频
    # Video synthesis
    # entry.merge_audio_video()


    # return (entry.txt, entry.converted_txt, entry.result_video_name)
    return (None, None, None)

def text_to_music_generate(caption: str, music_duration: int, music_gen: MusicGenerator, 
                           output_folder=Path("./outputs"), file_name=None):
    """
    Text to music generation as a base model
    """
    entry = Entry(image_recog=None, addtxt=None, music_gen=music_gen, music_duration=music_duration, output_folder=output_folder)
    entry.init_txt(caption)
    entry.txt_converter()
    entry.txt2music()
    entry.save_to_file(file_name=file_name)

    return (entry.txt, entry.converted_txt, entry.result_file_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mozart\'s Touch: Multi-modal Music Generation Framework')

    parser.add_argument('-d', '--test', help='Test mode', default=False, action='store_true')
    parser.add_argument('--text', help='Text-to-music', default=False, action='store_true')
    parser.add_argument('--image', help='Image-to-music', default=False, action='store_true')
    parser.add_argument('--image_file', help='Path to input image', default=None, type=str)
    parser.add_argument('--video', help='Video-to-music', default=False, action='store_true')
    parser.add_argument('--video_file', help='Path to input video', default=None, type=str)
    parser.add_argument('--output_filename', help='Output filename', default=None, type=str)
    parser.add_argument('--muvideo', help="Run generation on MUVideo dataset", default=False, action='store_true')
    parser.add_argument('--imemnet', help="Run generation on IMEMNet dataset", default=False, action='store_true')
    parser.add_argument('--custom', help="Run generation on custom csv file", default=False, action='store_true')
    parser.add_argument('--output_folder', help='Output folder path', default=None, type=str)
    
    args = parser.parse_args()
    test_mode = args.test # When True, disables the img2txt feature to save resources, used for debugging the program # True时关闭img2txt功能，节省运行资源，用于调试程序 

    image_recog = import_ir()
    music_gen = import_music_generator()
    output_folder = module_path / "outputs"
    music_duration = 10
    addtxt = None

    if args.text:
        caption = "Ella found a stray puppy in the park, muddy and trembling. She brought him home, gave him a bath, and named him Button. That night, he curled up on her bed, tail wagging. For the first time in weeks, she smiled in her sleep."
        result = text_to_music_generate(caption, music_duration, music_gen, output_folder)
    elif args.image:
        output_folder = Path('/home2/faculty/ktiurina/composer/data/survey')
        if (args.output_filename is not None):
            output_filename = args.output_filename
        else:
            output_filename = 'demo.wav'
        if (args.image_file is not None):
            img = Image.open(Path(args.image_file))
        else:
            img = Image.open(module_path / "static" / "test.jpg")
        result = img_to_music_generate(img, music_duration, image_recog, music_gen, output_folder, addtxt, output_filename)
    elif args.video:
        output_folder = Path('/home2/faculty/ktiurina/composer/data/survey')
        if (args.output_filename is not None):
            output_filename = args.output_filename
        else:
            output_filename = 'demo.wav'
        if (args.video_file is not None):
            video_path = Path(args.video_file)
        else:
            video_path = module_path / "static" / "stone.mp4"
        result = video_to_music_generate(video_path, image_recog, music_gen, output_folder, addtxt, output_filename)
    elif args.custom:
        logger.info("Start generating for custom dataset")
        output_folder = Path(args.output_folder)
        custom_styles = '/home2/faculty/ktiurina/composer/data/custom/MusicTheory.csv'
        styles_df = pd.read_csv(custom_styles)
        music_duration = 30
        os.makedirs(output_folder, exist_ok=True)
        for index, row in styles_df.iterrows():
            file_id = row['musicTheoryTerm'].replace("/", "_")
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                print(f"Sample {file_id} already generated. Skip")
            else:
                print(row['prompt'])
                result = text_to_music_generate(row['prompt'], music_duration, music_gen, output_folder, output_filename)
                print(f"Generated {output_filename}")
    elif args.imemnet:
        logger.info("Start generating for IMEMNet dataset")
        images_folder = '/home2/faculty/ktiurina/composer/data/IMEMNet/images_processed'
        output_folder = Path('/home2/faculty/ktiurina/composer/data/generated/IMEMNet/MozartsTouch')
        os.makedirs(output_folder, exist_ok=True)
        jpg_files = [f for f in os.listdir(images_folder) if f.lower().endswith('.jpg')]
        music_duration = 30
        for input_image in jpg_files:
            file_id = input_image.split('.')[0]
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
              print(f"Sample {file_id} already generated. Skip")
            else:
                image_path = os.path.join(images_folder, input_image)
                img = Image.open(Path(image_path))
                prompt = 'Generate music for the image'
                result = img_to_music_generate(img, music_duration, image_recog, music_gen, output_folder, prompt, output_filename)
                print(f"Generated {output_filename}")
    elif args.muvideo:
        logger.info("Start generating for MUVideo dataset")
        muvideo_instructions = '/home2/faculty/ktiurina/composer/data/MUVideo/MUVideoInstructions.json'
        videos_folder = '/home2/faculty/ktiurina/composer/data/MUVideo/muvideo_videos/hpctmp/e0589920/MUGen/data/MUVideo/audioset_video'
        with open(muvideo_instructions, 'r') as file:
            MuVideo = json.load(file)
        output_folder = '/home2/faculty/ktiurina/composer/data/generated/MUVideo/MozartsTouch'
        os.makedirs(output_folder, exist_ok=True)
        output_folder = Path(output_folder)
        for sample in MuVideo:
            file_id = sample['input_file'].split('.')[0]
            output_filename = f'{file_id}.wav'
            outfile = os.path.join(output_folder, f'{file_id}.wav')
            if os.path.isfile(outfile):
                logger.info(f"Sample {file_id} already generated. Skip")
            else:
                logger.info(f"Start generating for {file_id}")
                video_path = os.path.join(videos_folder, sample['input_file'])
                human_msg = [msg for msg in sample['conversation'] if msg.get('from') == 'human']
                if len(human_msg) > 0:
                    prompt = human_msg[0]['value']
                else:
                    prompt = 'Generate music for the video'
                try:
                    video_to_music_generate(Path(video_path), image_recog, music_gen, output_folder, addtxt, output_filename)
                    logger.info(f"Generated {output_filename}")
                except Exception as e:
                    logger.info('ERROR: ', e)
                    logging.exception("Error while generating music for MUVideo")

    else:
        raise TypeError("Select generation modality: --text, --image, --video.")

    # key_names = ("prompt", "converted_prompt", "result_file_name")

    # result_dict =  {key: value for key, value in zip(key_names, result)}

    # logger.info(result_dict)
    logger.info("Done.")
