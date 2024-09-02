from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
import threading

# Import the Realtime_Whisper class from your existing code
from whisper_realtime import Realtime_Whisper

app = FastAPI()

# Global variable to store the Realtime_Whisper instance
asr_instance = None

class TranscriptionResult(BaseModel):
    id: int
    start: float
    end: float
    text: str

@app.post("/start")
async def start_asr():
    global asr_instance
    if asr_instance is not None:
        raise HTTPException(status_code=400, detail="ASR is already running")
    
    asr_instance = Realtime_Whisper("tiny")
    asr_instance.start()
    return {"message": "ASR started successfully"}

@app.post("/stop")
async def stop_asr():
    global asr_instance
    if asr_instance is None:
        raise HTTPException(status_code=400, detail="ASR is not running")
    
    asr_instance.stop()
    asr_instance = None
    return {"message": "ASR stopped successfully"}

@app.get("/transcribe", response_model=List[TranscriptionResult])
async def get_transcription():
    global asr_instance
    if asr_instance is None:
        raise HTTPException(status_code=400, detail="ASR is not running")
    
    try:
        last_result = asr_instance.get_last_text()
        return [
            TranscriptionResult(
                id=segment['id'],
                start=round(segment['start'], 1),
                end=round(segment['end'], 1),
                text=segment['text']
            )
            for segment in last_result
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting transcription: {str(e)}")

@app.get("/status")
async def get_status():
    global asr_instance
    return {"status": "running" if asr_instance is not None else "stopped"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)