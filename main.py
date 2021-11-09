import logging
import os
from functools import reduce
from logging import getLogger

import click
import cv2
import numpy
import numpy as np
import pkg_resources
import pytesseract
from autocorrect import Speller
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
from symspellpy import SymSpell

logger = getLogger(__name__)
speller = Speller()
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


def get_file_extension(path):
    return os.path.splitext(path)[1]


def read_images(path):
    log("Reading images from path: %s", path)
    ext = get_file_extension(path)
    if (ext == ".jpg") or (ext == ".png"):
        return [Image.open(path)]
    elif ext == ".pdf":
        return convert_from_path(path)
    else:
        raise OcrError("Unsupported file type: {0}".format(ext))


class OcrError(Exception):
    pass


def preprocess_image(image):
    logger.info("Preprocessing image...")
    image = numpy.array(image)
    image = turn_image_to_grayscale(image)
    image = resize_image(image)
    image = invert(image)
    image = remove_horizontal_lines(image)
    image = remove_noise(image)
    image = Image.fromarray(image)
    return image


def turn_image_to_grayscale(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2GRAY)
        log("Turning image to grayscale...")
    return image


def resize_image(image):
    log("Resizing image...")
    return cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)


def binarize_image(image):
    log("Binarizing image...")
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 20)


def remove_noise(image):
    log("Removing noise from image...")
    return cv2.medianBlur(image, 3)


def invert(image):
    log("Inverting image...")
    return 255 - image


def remove_horizontal_lines(image):
    log("Removing horizontal lines from image")
    kernel = np.ones((1, 40), np.uint8)
    morphed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.subtract(image, morphed)
    return image


def post_process(text):
    log("Post processing recognized text: {0}".format(text))
    return spell_fix_text(text)


def spell_fix_text(text):
    log("Fixing text: %s", text)
    lines = map(lambda line: spell_fix_line(line), text.splitlines())
    return reduce(lambda x, y: x + os.linesep + y, lines)


def spell_fix_line(line):
    suggestions = sym_spell.lookup_compound(line,
                                            max_edit_distance=2,
                                            ignore_non_words=True,
                                            transfer_casing=True)
    if len(suggestions) > 0:
        result = suggestions[0].term
        log("Fixed line: %s Got result: %s", line, result)
        return result
    return ""


def log(message, *args, **kwargs):
    logger.debug(message, *args, **kwargs)


@click.command()
@click.option("--input", 'input_path', help="Input file name. Can be jpeg, png, pdf", prompt=True)
@click.option("--output", help="Output file name", prompt=True)
@click.option("--verbose", is_flag=True, default=False)
def main(input_path, output, verbose):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    try:
        images = read_images(input_path)
        images = map(lambda x: preprocess_image(x), images)
        texts = map(lambda x: pytesseract.image_to_string(x), images)
        result = post_process(reduce(lambda x, y: x + y, texts))
        with open(output, "w") as output_file:
            output_file.write(result)
        log("Written output result to file: %s", output)
    except OSError as e:
        print(e)
    except PDFPageCountError as e:
        print(e)


if __name__ == '__main__':
    main()
