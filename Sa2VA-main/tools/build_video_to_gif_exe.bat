@echo off
setlocal

cd /d %~dp0\..

echo [1/3] Install build dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :error
python -m pip install moviepy imageio imageio-ffmpeg pyinstaller
if errorlevel 1 goto :error

echo [2/3] Build EXE with PyInstaller...
pyinstaller ^
  --noconfirm ^
  --clean ^
  --onefile ^
  --windowed ^
  --name VideoToGif ^
  --collect-all moviepy ^
  --collect-all imageio ^
  --collect-all imageio_ffmpeg ^
  --hidden-import=imageio.plugins.ffmpeg ^
  tools\video_to_gif_gui.py
if errorlevel 1 goto :error

echo [3/3] Build success.
echo EXE output: dist\VideoToGif.exe
pause
exit /b 0

:error
echo Build failed. Please check error logs above.
pause
exit /b 1
