import os
import random
from PIL import Image, ImageDraw, ImageFont

# Generate simple images with rectangles/circles representing objects and save JSON annotations

CLASS_NAMES = ['box', 'person', 'face', 'earth', 'satellite']


def generate_sample(dataset_dir, n_images=20, size=(256, 256)):
    os.makedirs(dataset_dir, exist_ok=True)
    for i in range(n_images):
        img = Image.new('RGB', size, (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        draw = ImageDraw.Draw(img)
        # random number of objects
        n = random.randint(1, 3)
        anns = []
        for _ in range(n):
            w = random.randint(20, size[0]//2)
            h = random.randint(20, size[1]//2)
            x = random.randint(0, size[0]-w-1)
            y = random.randint(0, size[1]-h-1)
            cls = random.choice(CLASS_NAMES)
            draw.rectangle([x, y, x+w, y+h], outline=(255,255,255), width=2)
            draw.text((x+2, y+2), cls, fill=(255,255,255))
            anns.append({
                'label': cls,
                'object_name': cls,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            })
        path = os.path.join(dataset_dir, f'image_{i:04d}.png')
        img.save(path)
        # save annotation JSON
        j = {'image': os.path.basename(path), 'annotations': anns}
        with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(j, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    generate_sample('data/sample_dataset', n_images=50)
