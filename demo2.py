import os
import sys
import tqdm
import urllib.request
import datetime
import whisper
import whisper.transcribe

def format_vtt_time(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

class _CustomProgressBar(tqdm.tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current = self.n  # Set the initial value

    def update(self, n):
        super().update(n)
        self._current += n

        # Handle progress here
        print("Progress: " + str(self._current) + "/" + str(self.total))


# Inject into tqdm.tqdm of Whisper, so we can see progress
transcribe_module = sys.modules['whisper.transcribe']
transcribe_module.tqdm.tqdm = _CustomProgressBar

model = whisper.load_model("medium")
print("Finished load model")
print("Transcribing")
result = model.transcribe("Untitled.mp4", language='en', fp16=False)
print(result['text'])
save_target = 'greedy2.vtt'
with open(save_target, 'w') as file:
    for indx, segment in enumerate(result['segments']):
        file.write(str(indx + 1) + '\n')
        file.write(format_vtt_time(segment['start']) + ' --> ' + format_vtt_time(segment['end']) + '\n')
        file.write(segment['text'].strip() + '\n')
        file.write('\n')
print("Finished")
