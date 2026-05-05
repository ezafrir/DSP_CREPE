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







print("Replacing WORLD's pitch estimate with CREPE's...")
 
# CREPE's time grid 
time_crepe_grid = np.arange(len(f0_crepe)) * (512 / sr)
 
f0_crepe_interp = np.interp(t_world, time_crepe_grid, f0_crepe)
 
# Only replace frames where BOTH CREPE and WORLD agree there's singing
voiced_both = (f0_crepe_interp > 0) & (f0_world > 0)
f0_combined = f0_world.copy()
f0_combined[voiced_both] = f0_crepe_interp[voiced_both]
 
print(f"  Replaced {np.sum(voiced_both)} voiced frames with CREPE estimates\n")
 



#here we are shifting the pitch to hopefully have a harmony!!

SEMITONES = 7  # perfect 5th- apparently this sounds the best

ratio = 2 ** (SEMITONES / 12.0)
print(f"Step 5: Shifting pitch by {SEMITONES} semitones (perfect 5th)")
print(f"  Frequency ratio: {ratio:.4f}x")
print(f"  e.g. if Papa sings 220 Hz -> harmony is {220 * ratio:.1f} Hz\n")
 
f0_harmony = f0_combined.copy()
f0_harmony[f0_harmony > 0] *= ratio  # only shift voiced frames


#now we put them all back together...
print("Step 6: Resynthesizing harmony voice with WORLD...")
harmony = pw.synthesize(
    f0_harmony.astype(np.float64),
    sp.astype(np.float64),
    ap.astype(np.float64),
    sr
).astype(np.float32)
print(f"  Synthesized {len(harmony)/sr:.2f} seconds of harmony audio\n")




#adding the two signals together and save. 
# we turn down the volume of the harmony so it doesnt sound too messy....