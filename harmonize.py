import numpy as np
import soundfile as sf
import  librosa
import crepe
import pyworld as pw

#librosa.load() reads the .wav file and returns audio in an array & sample rate
print("Loading audio...")
audio, sr = librosa.load("papas_magalitriti.wav", sr=22050, mono=True)
 # 22050 samples per second
duration = len(audio) / sr
print(f"  Loaded! {duration:.2f} seconds of audio at {sr} Hz")
print(f"  That's {len(audio):,} individual samples\n")

print("Extracting pitch with CREPE...")
print("  (First run will download model weights...)") #for neural net
time_crepe, f0_crepe, confidence, _ = crepe.predict(audio, sr, viterbi=True, verbose=1)
# time_crepe is array of timestamps
# f0 is array of freqs in hz at each timestamp
# confidence is how sure CREPE is of either silence or noise i believe
# ^ so we will zero out frames where CREPE isn't confident

f0_crepe[confidence < 0.5] = 0.0
 
voiced_frames = np.sum(f0_crepe > 0)
total_frames = len(f0_crepe)
print(f"  Found pitch in {voiced_frames}/{total_frames} frames")
print(f"  Pitch range: {f0_crepe[f0_crepe > 0].min():.1f} Hz "
      f"— {f0_crepe[f0_crepe > 0].max():.1f} Hz\n")

print("Decomposing audio with WORLD vocoder...")
print("  Separating: pitch | timbre | breathiness")
#vocoder separates voice signal into individual components to manipuulate



x = audio.astype(np.float64)
 
# dio() = Distributed Inline-filter Operation
#estimates fundamental freq f0 fast
f0_world, t_world = pw.dio(x, sr)
 
# stonemask() refines the rough dio() estimate
f0_world = pw.stonemask(x, f0_world, t_world, sr)
 
# cheaptrick() extracts the spectral envelope (timbre)
#basically the shape of the voice and what makes someone sound like themselves
sp = pw.cheaptrick(x, f0_world, t_world, sr)
 
# d4c() extracts aperiodicity (how noisy each frame is)
ap = pw.d4c(x, f0_world, t_world, sr)
 
print(f"  Decomposed into {len(f0_world)} frames\n")







