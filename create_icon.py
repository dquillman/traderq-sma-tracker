"""Create a simple icon for TraderQ app"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create a 256x256 image with a gradient background
size = 256
img = Image.new('RGB', (size, size), color='#1a1a2e')

# Create a drawing context
draw = ImageDraw.Draw(img)

# Draw a modern background with gradient effect
for i in range(size):
    color_value = int(26 + (i / size) * 30)
    draw.rectangle([(0, i), (size, i+1)], fill=(color_value, color_value, color_value + 20))

# Draw stylized "TQ" text
try:
    # Try to use a system font
    font = ImageFont.truetype("arial.ttf", 120)
    font_small = ImageFont.truetype("arial.ttf", 40)
except:
    # Fallback to default font
    font = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Draw "TQ" in the center with a nice color
text = "TQ"
# Get text bounding box
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]
x = (size - text_width) // 2
y = (size - text_height) // 2 - 20

# Draw shadow
draw.text((x+3, y+3), text, fill='#000000', font=font)
# Draw main text with gradient effect
draw.text((x, y), text, fill='#17c964', font=font)

# Draw subtitle
subtitle = "Tracker"
bbox_sub = draw.textbbox((0, 0), subtitle, font=font_small)
sub_width = bbox_sub[2] - bbox_sub[0]
x_sub = (size - sub_width) // 2
y_sub = y + text_height + 10
draw.text((x_sub, y_sub), subtitle, fill='#888888', font=font_small)

# Draw a simple chart line pattern at the bottom
points = []
for i in range(0, size, 20):
    import math
    y_val = size - 40 + math.sin(i / 20) * 15
    points.append((i, y_val))

for i in range(len(points) - 1):
    draw.line([points[i], points[i+1]], fill='#17c964', width=3)

# Save as ICO file
ico_path = os.path.join(os.path.dirname(__file__), 'traderq_icon.ico')
img.save(ico_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])

print(f"Icon created: {ico_path}")
