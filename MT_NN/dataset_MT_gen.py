import pygame
import json
import os
from random import randint, uniform

pygame.init()
pygame.display.set_mode((1, 1))

train_data = {'total': 2000, 'dir': 'train_mt'}
test_data = {'total': 100, 'dir': 'test_mt'}
total_bk = 14
total_cls = 4
dir_out = ''
cls = [(255, 255, 255), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
file_format = 'format_mt.json'

if not os.path.exists(dir_out):
    os.makedirs(dir_out)
os.makedirs(os.path.join(dir_out, 'train_mt'), exist_ok=True)
os.makedirs(os.path.join(dir_out, 'test_mt'), exist_ok=True)

Rudeus = [pygame.image.load(f'ваш путь к папке с фото{i}.png').convert_alpha()
          for i in range(9)]
Eris = [pygame.image.load(f'ваш путь к папке с фото{i}.png').convert_alpha()
        for i in range(10)]
Sylphiette = [pygame.image.load(f'ваш путь к папке с фото{i}.png').convert_alpha()
              for i in range(15)]

characters = {
    0: Rudeus,
    1: Eris,
    2: Sylphiette
}

backs = [pygame.image.load(f'ваш путь к папке с фото для фона{n}.jpeg') for n in range(total_bk)]


def augment_char(char_img):
    img = char_img.copy()

    if randint(0, 1) == 1:
        img = pygame.transform.flip(img, True, False)

    angle = uniform(-25, 25)
    img = pygame.transform.rotate(img, angle)

    scale = uniform(0.8, 1.2)
    w, h = img.get_size()
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = pygame.transform.scale(img, (new_w, new_h))

    max_size = 240
    w, h = img.get_size()
    if w > max_size or h > max_size:
        scale_down = max_size / max(w, h)
        new_w = max(1, int(w * scale_down))
        new_h = max(1, int(h * scale_down))
        img = pygame.transform.scale(img, (new_w, new_h))

    return img


for info in (train_data, test_data):
    labels = {}

    for i in range(1, info['total'] + 1):
        file_out = f'character_{i}.png'
        im = backs[randint(0, total_bk - 1)].copy()
        bg_w, bg_h = im.get_size()

        # шум
        for _ in range(randint(20, 100)):
            x0 = randint(0, bg_w - 1)
            y0 = randint(0, bg_h - 1)
            pygame.draw.circle(im, cls[randint(0, total_cls - 1)], (x0, y0), 1)

        character_class = randint(0, 2)
        variants = characters[character_class]
        character_img = variants[randint(0, len(variants) - 1)]
        character_img = augment_char(character_img)

        w, h = character_img.get_size()
        x = randint(0, bg_w - w) if bg_w > w else 0
        y = randint(0, bg_h - h) if bg_h > h else 0
        im.blit(character_img, (x, y))

        labels[file_out] = character_class

        pygame.image.save(im, os.path.join(dir_out, info['dir'], file_out))

    with open(os.path.join(dir_out, info['dir'], file_format), 'w') as fp:
        json.dump(labels, fp)

