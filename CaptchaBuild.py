from captcha.image import ImageCaptcha
from PIL import Image
from random import randint
import threading


def build(text, name):
    image = ImageCaptcha()
    captcha = image.generate(text)
    captcha_image = Image.open(captcha)
    captcha_image.save("train/" + name + ".png")


def pick(size, base, name):
    length = len(base) - 1
    base_out = ""
    pick_out = ""
    for x in range(size):
        idx = randint(0, length)
        base_out += base[idx]
        pick_out += name[idx]
    return base_out, pick_out


def run():
    build(*pick(4, all_text_list, name_text_list))


if __name__ == '__main__':
    process = []
    num_list = [str(x) for x in range(0, 10)]
    lower_list = [chr(x) for x in range(97, 123)]
    upper_list = [chr(x) for x in range(65, 91)]
    all_text_list = num_list + lower_list + upper_list
    name_text_list = num_list + lower_list + lower_list
    for x in range(10000):
        t = threading.Thread(target=run)
        process.append(t)
        t.start()
    for x in process:
        x.join()
