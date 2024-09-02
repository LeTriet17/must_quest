import gradio as gr
import requests
from typing import List, Dict
import time

# FastAPI server URL
API_URL = "http://localhost:8000"

class ASRManager:
     def __init__(self):
          self.is_running = False
          self.transcription_history: List[Dict] = []

     def start_asr(self):
          if not self.is_running:
               response = requests.post(f"{API_URL}/start")
               if response.status_code == 200:
                    self.is_running = True
                    return "ASR started successfully"
               else:
                    return f"Failed to start ASR: {response.json().get('detail', 'Unknown error')}"
          return "ASR is already running"

     def stop_asr(self):
          if self.is_running:
               response = requests.post(f"{API_URL}/stop")
               if response.status_code == 200:
                    self.is_running = False
                    return "ASR stopped successfully"
               else:
                    return f"Failed to stop ASR: {response.json().get('detail', 'Unknown error')}"
          return "ASR is not running"

     def get_transcription(self):
          if not self.is_running:
               return "ASR is not running"
          
          try:
               response = requests.get(f"{API_URL}/transcribe")
               if response.status_code == 200:
                    new_segments = response.json()
                    for segment in new_segments:
                         if segment not in self.transcription_history:
                              self.transcription_history.append(segment)
                    return self.format_transcription()
               else:
                    return f"Error getting transcription: {response.json().get('detail', 'Unknown error')}"
          except Exception as e:
               return f"Error getting transcription: {str(e)}"

     def format_transcription(self):
          return "\n".join([f"{s['text']}" for s in self.transcription_history])

asr_manager = ASRManager()

def update_transcription():
    while True:
        if asr_manager.is_running:
             
            new_transcription = asr_manager.get_transcription()
            yield new_transcription
        time.sleep(2)  # Wait for 2 seconds before the next update

with gr.Blocks() as demo:
     gr.Markdown("# Realtime Whisper ASR Demo")
     
     with gr.Row():
          start_button = gr.Button("Start ASR")
          stop_button = gr.Button("Stop ASR")
     
     status_text = gr.Textbox(label="Status")
     transcription_box = gr.TextArea(
          label="Transcription", 
          interactive=False,
          value="""Cảm ơn các bạn\nChào tất cả!"""
     )

     start_button.click(
          fn=asr_manager.start_asr,
          outputs=status_text
     )

     stop_button.click(
          fn=asr_manager.stop_asr,
          outputs=status_text
     )
     

if __name__ == "__main__":
    demo.queue().launch()