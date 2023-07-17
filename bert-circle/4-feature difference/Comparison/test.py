import matplotlib.font_manager as fm

# 要查找的字体名称
font_name = "Arial Monospaced MT"

# 查找字体文件路径
font_path = fm.findfont(font_name)

if font_path is not None:
    print(f"The font '{font_name}' is installed at: {font_path}")
else:
    print(f"The font '{font_name}' is not installed.")
