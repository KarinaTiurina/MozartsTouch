from openai import OpenAI
from pathlib import Path
import httpx
import yaml
from loguru import logger

module_path = Path(__file__).resolve().parent.parent
with open(module_path / 'config.yaml', 'r', encoding='utf8') as file:
    config = yaml.safe_load(file)

class TxtConverter:
    def __init__(self):
        self.use_llm = config.get('USE_LLM', False)
        if self.use_llm:
            self.model = config['DEFAULT_LLM_MODEL']
            self.api_url = config['LLM_MODEL_CONFIG']['API_BASE_URL']
            self.api_key = config['LLM_MODEL_CONFIG']['API_KEY'] or self._prompt_for_api_key()
            self.client = OpenAI(
                base_url=self.api_url, 
                api_key=self.api_key,
                http_client=httpx.Client(base_url=self.api_url, follow_redirects=True),
            )

    def _prompt_for_api_key(self):
        api_key = input("Enter your OpenAI API key: ")
        config['LLM_MODEL_CONFIG']['API_KEY'] = api_key
        with open(module_path / 'config.yaml', 'w') as file:
            yaml.dump(config, file)
        return api_key

    def process_video_description(self, texts: list):
        if not self.use_llm:
            return str(texts[0])
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are about to process a sequence of captions, each corresponding to a distinct frame sampled from a video. Your task is to convert these captions into a cohesive, well-structured paragraph. This paragraph should describe the video in a fluid, engaging manner and follows these guidelines: avoiding semantic repetition to the greatest extent, and giving a description in less than 200 characters."},
                {"role": "user", "content": str(texts)}
            ]
        )
        result = completion.choices[0].message.content
        return result

    def txt_converter(self, content, addtxt=None):
        if addtxt:
            content += addtxt #在这里加入附加文本然后一起丢进llm跑
        # logger.info("filtered_prompt result:"+content.encode('utf8', errors='replace').decode('utf8'))

        if not self.use_llm:
            return content
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Convert in less than 200 characters this image caption to a very concise musical description with musical terms, so that it can be used as a prompt to generate music through AI model, strictly in English. You need to speculate the mood of the given image caption and add it to the music description. You also need to specify a music genre in the description such as pop, hip hop, funk, electronic, jazz, rock, metal, soul, R&B etc."},
                {"role": "user", "content": "a city with a tower and a castle in the background, a detailed matte painting, art nouveau, epic cinematic painting, kingslanding"},
                {"role": "assistant", "content": "A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle."},
                {"role": "user", "content": "a group of people sitting on a beach next to a body of water, tourist destination, hawaii"},
                {"role": "assistant", "content": "Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach."},
                {"role": "user", "content": content}
            ]
        )
        converted_result = completion.choices[0].message.content
        logger.info("converted result: " + converted_result.encode('utf8', errors='replace').decode('utf8'))
        return converted_result

    def video_txt_converter(self, content, addtxt=None):
        if addtxt:
            content += addtxt
        if not self.use_llm:
            logger.info(type(content))
            return content
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Convert in less than 200 characters this video caption to a very concise musical description with musical terms, so that it can be used as a prompt to generate music through AI model, strictly in English. You need to speculate the mood of the given video caption and add it to the music description. You also need to specify a music genre in the description such as pop, hip hop, funk, electronic, jazz, rock, metal, soul, R&B etc."},
                {"role": "user", "content": "Two men playing cellos in a room with a piano and a grand glass window backdrop."},
                {"role": "assistant", "content": "Classical chamber music piece featuring cello duet, intricate piano accompaniment, emotive melodies set in an elegant setting, showcasing intricate melodies and emotional depth, the rich harmonies blend seamlessly in an elegant and refined setting, creating a symphonic masterpiece."},
                {"role": "user", "content": "A man with guitar in hand, captivates a large audience on stage at a concert. The crowd watches in awe as the performer delivers a stellar musical performance."},
                {"role": "assistant", "content": "Rock concert with dynamic guitar riffs, precise drumming, and powerful vocals, creating a captivating and electrifying atmosphere, uniting the audience in excitement and musical euphoria."},
                {"role": "user", "content": content}
            ]
        )
        converted_result = completion.choices[0].message.content
        logger.info("converted result: " + converted_result.encode('utf8', errors='replace').decode('utf8'))
        return converted_result

if __name__ == "__main__":
    # content = "a wreath hanging from a rope, an album cover inspired, land art, japanese shibari with flowers, hanging from a tree,the empress’ hanging"
    content = input()
    txt_con = TxtConverter()
    converted_result = txt_con.txt_converter(content)