import datetime
import whisper
from tqdm import tqdm

model = whisper.load_model('base.en')
option = whisper.DecodingOptions(language='en', fp16=False)
result = model.transcribe('greedy.mp4')
print(result)


def format_vtt_time(seconds):
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


save_target = 'greedy.vtt'
with open(save_target, 'w') as file:
    for indx, segment in enumerate(result['segments']):
        file.write(str(indx + 1) + '\n')
        file.write(format_vtt_time(segment['start']) + ' --> ' + format_vtt_time(segment['end']) + '\n')
        file.write(segment['text'].strip() + '\n')
        file.write('\n')

print("Finished")
