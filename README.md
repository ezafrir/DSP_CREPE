# DSP_CREPE
Harmony Synthesizer!
Extracts pitch from a singing WAV file using CREPE and resynthesizes a harmony voice using the WORLD vocoder.

## Setup

```bash
python3 -m venv harmony-env
source harmony-env/bin/activate
pip install --upgrade pip setuptools wheel
pip install pyworld numpy scipy soundfile librosa
```

### Install CREPE manually (required due to Python 3.12 compatibility issue :( )

```bash
git clone https://github.com/marl/crepe.git
cd crepe
```

Edit `setup.py`... replace the `install_requires` block with:
```python
install_requires=open(
    os.path.join(os.path.dirname(__file__), "requirements.txt")
).read().splitlines(),
```
Also delete the `import pkg_resources` line at the top.

Then install:
```bash
pip install .
cd ..
```

### Fix pyworld (required due to Python 3.12 compatibility issue)

Open `harmony-env/lib/python3.12/site-packages/pyworld/__init__.py` and:
- Delete the `import pkg_resources` line
- Replace `__version__ = pkg_resources.get_distribution('pyworld').version` with `__version__ = "0.3.0"`

## Run

Place your WAV file in the `DSP_CREPE` folder, then:

```bash
source harmony-env/bin/activate
python harmonize.py
```

Output will be saved as `papas_megalitriti_harmony.wav` in the same folder.

## Configuration

In `harmonize.py`, change the `SEMITONES` variable to try different intervals!!

| Value | Interval |
|-------|----------|
| 4 | Major 3rd  (default) |
| 7 | Perfect 5th |
| 12 | Octave above |
| -5 | Perfect 4th below |
| -12 | Octave below |
