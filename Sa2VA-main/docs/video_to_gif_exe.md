# Video To GIF EXE 工具

这个工具支持：

- 选择本地视频文件作为输入
- 指定本地 GIF 输出路径
- 设置 GIF 帧率（FPS）和最大宽度（可选）

## 源码位置

- GUI 脚本：`tools/video_to_gif_gui.py`
- 打包脚本（Windows）：`tools/build_video_to_gif_exe.bat`

## 在 Windows 打包 EXE

在项目根目录打开 `cmd`，执行：

```bat
tools\build_video_to_gif_exe.bat
```

打包成功后，输出文件为：

```text
dist\VideoToGif.exe
```

## 直接运行 Python 版本（可选）

先安装依赖：

```bash
python -m pip install moviepy imageio imageio-ffmpeg
```

再运行：

```bash
python tools/video_to_gif_gui.py
```

## 使用说明

1. 点击 `Browse` 选择输入视频。
2. 点击 `Browse` 选择输出 GIF 路径。
3. 设置 `FPS`（建议 `8-15`）。
4. 设置 `Max width`（例如 `480` 或 `640`）。
5. 点击 `Convert to GIF` 开始转换。

