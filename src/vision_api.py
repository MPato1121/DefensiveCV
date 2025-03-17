import openai
import cv2
import json
import base64
import requests
import numpy as np
from typing import List, Dict, Optional, Iterator, Tuple
from tqdm import tqdm
import supervision as sv

# Placeholder for API key (replace when ready)
API_KEY = "YOUR_OPENAI_API_KEY"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

class VisionAPI:
    def __init__(self, api_key=API_KEY):
        self.api_key = api_key
    
    def encode_image_to_base64(self, image: np.ndarray) -> str:
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Could not encode image to JPEG format.")
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return encoded_image
    
    def compose_payload(self, images: List[np.ndarray], prompt: str) -> dict:
        text_content = {"type": "text", "text": prompt}
        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encode_image_to_base64(image=image)}"
                }
            }
            for image in images
        ]
        return {
            "model": "gpt-4-vision-preview",
            "messages": [
                {"role": "user", "content": [text_content] + image_content}
            ],
            "max_tokens": 300
        }
    
    def compose_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def prompt_image(self, images: List[np.ndarray], prompt: str) -> str:
        headers = self.compose_headers()
        payload = self.compose_payload(images, prompt)
        response = requests.post(url=OPENAI_API_URL, headers=headers, json=payload).json()

        if 'error' in response:
            raise ValueError(response['error']['message'])
        return response['choices'][0]['message']['content']
    
    def update_class_id(self, data: Dict[str, str], detections: sv.Detections, classes: List[str]) -> sv.Detections:
        result = detections.copy()
        category_to_id = {category: i for i, category in enumerate(classes)}
        mapped_data = {int(k): category_to_id.get(v) for k, v in data.items() if v in category_to_id}
        result.class_id = np.array([mapped_data.get(i, None) for i in range(len(result))])
        return result

    def generate_prompt(self, team_colors: Tuple[str, str]) -> str:
        team1, team2 = team_colors
        prompt = (
            f"Identify the team affiliation of the marked individual in the image: "
            f"Options are `{team1}` or `{team2}`. If the marked individual does not belong to either team "
            f"(e.g., is a referee, coach, or fan), return `none`. Referees are distinguishable by their black uniforms. "
            f"Coaches and fans should not be considered as team members. "
            f"Provide the results in JSON format. "
            f"Format the output like this: {{'0': '{team1}', '1': '{team2}', ...}} "
            f"Use double quotes to enclose property names. "
            f"Do not surround the result with backticks (`)"
            f"notalk;justgo"
        )
        return prompt
    
    def analyze_frame(self, frame, players, team_colors: Tuple[str, str]):
        crops = [sv.crop_image(frame, players.xyxy[i]) for i in range(len(players))]
        prompt = self.generate_prompt(team_colors)
        response = self.prompt_image(crops, prompt)
        return json.loads(response)
    
    def process_batches(self, frame, players, team_colors: Tuple[str, str], batch_size=5):
        xyxy_batches = list(self.chunk_list(players.xyxy, batch_size))
        merged_response = {}
        for i, batch in enumerate(xyxy_batches):
            start_index = i * batch_size
            prompt = self.generate_prompt(team_colors)
            response = self.analyze_frame(frame, players, team_colors)
            for key, value in response.items():
                merged_response[str(int(key) + start_index)] = value
        return merged_response

    @staticmethod
    def chunk_list(lst: List, n: int) -> Iterator[List]:
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

# Example usage:
if __name__ == "__main__":
    api = VisionAPI()

    # Example frame and player detection data
    frame = cv2.imread("example.jpg")
    players = sv.Detections.from_ultralytics(...)  # Replace with actual detection results
    
    # Example dynamic team colors
    team_colors = ("blue", "red")

    response = api.process_batches(frame, players, team_colors)
    print(response)
