# Darktable Film Scanning Plugins

Lua plugins for darktable to automate DSLR film scanning workflow.

## Features

- **AutoCrop** (`auto_crop/`) — Detect film holder edges and auto-crop scanned frames
- **AutoRetouch** (`auto_retouch/`) — Detect dust spots and auto-apply healing brushes

Both use a two-component architecture: Lua plugin (darktable integration) + Python script (image analysis via OpenCV).

## Installation

### 1. Install the Lua Plugins

Copy this directory to your darktable Lua scripts directory:

**Windows:**
```
%LOCALAPPDATA%\darktable\lua\
```

**macOS/Linux:**
```
~/.config/darktable/lua/
```

Enable the scripts in darktable:
1. Open darktable
2. Go to **Settings → Lua Scripts**
3. Enable **AutoCrop** and/or **AutoRetouch**

### 2. Set Up Python Environment

You must have **conda** (Anaconda or Miniconda) installed.

#### Create the conda environment:

Navigate to the plugin directory and run:

```bash
conda env create -f environment.yml
```

This creates a conda environment named `autocrop` with all required dependencies.

#### Verify the environment:

```bash
conda activate autocrop
python --version
```

### 3. Test the Setup

1. Open darktable
2. Select one or more images in lighttable view
3. Run the AutoCrop action (via GUI or keyboard shortcut)
4. Check the console for export progress and Python processing output

## Files

- `auto_crop/auto_crop.lua` - AutoCrop darktable Lua plugin
- `auto_crop/auto_crop.py` - AutoCrop Python image analysis
- `auto_retouch/auto_retouch.lua` - AutoRetouch darktable Lua plugin
- `auto_retouch/detect_dust.py` - Dust detection Python script
- `auto_retouch/dust_detection_spec.md` - Dust detection algorithm specification
- `environment.yml` - Shared conda environment specification
- `CLAUDE.md` - Developer documentation

## Usage

### From darktable GUI

1. Select images in lighttable
2. Go to **Image → AutoCrop** (or use configured keyboard shortcut)
3. Images export to `%TEMP%\darktable_autocrop_[timestamp]\`
4. Python script processes the exported images

### Customization

Edit `process_images.py` to add your custom image processing logic. The script receives the export directory path as an argument.

## Conda Environment

The `autocrop` environment includes:
- Python 3.11
- Pillow (image processing)
- NumPy (numerical operations)
- OpenCV (computer vision)

To add more dependencies:

```bash
conda activate autocrop
conda install package-name
```

Then update `environment.yml`:

```bash
conda env export --name autocrop --from-history > environment.yml
```

## Updating

### Update the conda environment:

```bash
conda env update -f environment.yml --prune
```

### Reload Lua script in darktable:

1. Go to **Settings → Lua Scripts**
2. Disable and re-enable **AutoCrop**

Or restart darktable.

## Troubleshooting

### "conda: command not found"

Ensure conda is in your system PATH. On Windows, you may need to run from **Anaconda Prompt** or add conda to PATH.

### "Python script not found"

Ensure Python scripts are in the same directory as their Lua plugins:
```
%LOCALAPPDATA%\darktable\lua\ilya\auto_crop\
%LOCALAPPDATA%\darktable\lua\ilya\auto_retouch\
```

### "Environment autocrop not found"

Create the environment:
```bash
conda env create -f environment.yml
```

### Script not appearing in darktable

1. Check the Lua script is in the correct directory
2. Check for errors in darktable console (View → Console)
3. Ensure darktable API version is 7.0.0+ (darktable 4.x or later)

## License

GPL-3.0 or later (same as darktable)

## Author

Ilya Palopezhentsev
