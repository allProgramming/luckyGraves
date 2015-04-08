import requests
from requests.exceptions import MissingSchema, InvalidSchema
import numpy
import tesseract
import sys
import argparse
import collections
import math
import ntpath
import os
import string
import cv2
import cv2.cv as cv
import mysql.connector as mdb
from itertools import product
import unicodedata
from datetime import date

## BAD. To-do: make these class variables?
api = None
sql_conn = None
sql_cur = None
knownWords = []

## TEMPORARY. To-do: make these command-line parameters
CLEANER_HOST = ''
MYSQL_HOST = ''
MYSQL_DATABASE = ''
MYSQL_USER = ''
MYSQL_PASSWORD = ''

## Main function.
def main(argv = []):
    """ Overview of program execution.

    :Parameters:
      - `argv` (list) - command-line arguments

    :Returns:
      None
    """
    cmdParams = parseArgs(argv)
    initTools(cmdParams)
    for imagePath in cmdParams.input:
        imageID, rawImage = acquireImage(imagePath)
        cleanedImage = cv2.cvtColor(rawToCV2(cleanImage(rawImage)), cv2.COLOR_BGR2GRAY)
        variations = getVariations(cleanedImage)
        for sampleParams in variations:
            sampledImage = sampleImage(cleanedImage, sampleParams)
            if cmdParams.show:
                showImage(sampledImage, "Sampled Image")
            text = performOCR(sampledImage)
            storeText(imageID, text, sampleParams)
    shutdownTools()

## Parse command-line arguments.
def parseArgs(argv):
    """ Parse command-line arguments.

    :Parameters:
      - `argv` (list) - command-line arguments

    :Returns:
      Parsed command-line arguments

    :Return Type:
      dict
    """
    argParser = argparse.ArgumentParser(
        description="Sample headstone image(s), convert to text, and store in a database."
    )
    argParser.add_argument(
        '-i', '--input',
        help="URL, local file path, or directory of image(s)",
        type=str,
        required=True
    )
    argParser.add_argument(
        '-s', '--show',
        help="show sampled image and wait for key press",
        action="store_true"
    )
    cmdParams = argParser.parse_args(argv[1:])

    # Convert input into a list of 1+ files (depending on if it's a directory)
    if os.path.isdir(cmdParams.input):
        cmdParams.input = \
            [os.path.join(cmdParams.input, filename)
            for filename in os.listdir(cmdParams.input)
            if filename.endswith(".jpg")]
    else:
        cmdParams.input = [cmdParams.input]

    return cmdParams

def initTools(cmdParams):
    global api, sql_conn, sql_cur, knownWords

    api = tesseract.TessBaseAPI()
    api.Init(os.environ['TESSDATA_PREFIX'], "eng", tesseract.OEM_DEFAULT)
    api.SetPageSegMode(tesseract.PSM_AUTO_OSD)

    if MYSQL_HOST and MYSQL_DATABASE and MYSQL_USER and MYSQL_PASSWORD:
        sql_conn = mdb.connect(host=MYSQL_HOST,
                               database=MYSQL_DATABASE,
                               user=MYSQL_USER,
                               password=MYSQL_PASSWORD)
        sql_cur = sql_conn.cursor()

    # [To-do] Load known/recognizable words from a dictionary
    knownWords = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    knownWords += [str(year) for year in range(1000, date.today().year)]

## Acquire image for processing.
def acquireImage(imagePath):
    """ Acquire image for processing.

    :Parameters:
      - `imagePath` (string) - path/URL to file of image

    :Returns:
      - Image ID (string) - a hopefully unique identifier of the image
      - Image (cv2 image) - the loaded, unprocessed image
    """
    print "Acquiring an image"
    imageID = ntpath.splitext(ntpath.basename(imagePath))[0] # To-do: make more unique (yet still deterministic)
    try:
        # Verify that imagePath is a URL, and attempt to fetch it
        req = requests.get(imagePath, stream=True)
        req.raise_for_status()
        rawImage = req.raw
    except (MissingSchema, InvalidSchema) as e:
        # If imagePath is not a URL, then attempt to open the file locally
        rawImage = open(imagePath, 'rb')
    print "Acquired image: %s" % imageID
    return imageID, rawImage

## Clean up an image.
def cleanImage(image):
    """ Clean up an image.

    The intent here is to clean up the image prior to sampling.

    :Parameters:
      - `image` (cv2 image) - image to be cleaned up

    :Returns:
      The cleaned image

    :Return Type:
      cv2 image
    """
    print "Cleaning image"
    if not CLEANER_HOST:
        return image
    req = requests.post(
        'http://' + CLEANER_HOST + '/headstone-cleaner/rest/upload',
        files={'file': image}
    )
    req.raise_for_status()
    uploadedID = ntpath.basename(req.text)
    print "Original uploaded: %s" % uploadedID
    req = requests.post(
        'http://' + CLEANER_HOST + '/headstone-cleaner/rest/binarize',
        uploadedID
    )
    req.raise_for_status()
    zonedImagePath = \
        'http://' + CLEANER_HOST + '/headstone-cleaner/' + \
        req.json()['zonedImagePath']
    print "Original cleaned: %s" % zonedImagePath
    req = requests.get(zonedImagePath, stream=True)
    req.raise_for_status()
    print "Clean image handle acquired"
    return req.raw

## Convert raw image to cv2.
def rawToCV2(image):
    """ Convert raw image to cv2.

    Convert raw image (file-like object) into a cv2 image.

    :Parameters:
      - `image` (file-like object) - image to be converted

    :Returns:
      The constructed cv2 image

    :Return Type:
      cv2 image
    """
    imageArray = numpy.asarray(bytearray(image.read()), dtype=numpy.uint8)
    cv2Image = cv2.imdecode(imageArray, cv2.IMREAD_UNCHANGED)
    if cv2Image is None or len(cv2Image[0][0]) <= 3: # No alpha channel?
        return cv2Image
    ch1, ch2, ch3, alpha = cv2.split(cv2Image)
    alpha = 1 - (alpha / 255)
    ch1 = ch1 + (255 - ch1) * alpha
    ch2 = ch2 + (255 - ch2) * alpha
    ch3 = ch3 + (255 - ch3) * alpha
    return cv2.merge((ch1, ch2, ch3))


## Determine variations for sampling.
def getVariations(image):
    """ Determine variations for sampling.

    Generate a list of parameters to be applied during sampling.

    :Parameters:
      - `image` (cv2 image) - image to determine variations for

    :Returns:
      Parameters to apply during sampling

    :Return Type:
      list of dict's
    """
    combinations = {}

    # Determine if the image is horizontal or vertical
    edges = cv2.Canny(image, 80, 120)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 2, 100, None, 800, 20)
    needsRotate = 0
    allLines = []
    if not lines == None:
        allLines = lines[0]
    for line in allLines:
        needsRotate += \
            -1 if abs(line[0] - line[2]) > abs(line[1] - line[3]) else 1
    combinations['rotate'] = [90, 270] if needsRotate > 0 else [0, 180]

    combinations['blur'] = [7]

    threshold_step = 1
    combinations['threshold'] = [x * threshold_step for x in range(0, 255 / threshold_step)]

    # Return all permutations of items in "combinations"
    return [
        dict(zip(combinations, value))
        for value in product(*combinations.values())
    ]

## Sample the image.
def sampleImage(image, params):
    """ Sample the image.

    Apply thresholding, rotation, etc. as specified by the given parameters.

    :Parameters:
      - `image` (cv2 image) - image to be sampled
      - `params` (dict) - parameters to apply to the image

    :Returns:
      Sampled image

    :Return Type:
      cv2 image
    """
    print "Sampling image, with params: %s" % params
    sampledImage = image
    
    if params['rotate'] == 90:
        sampledImage = cv2.transpose(sampledImage)
        sampledImage = cv2.flip(sampledImage, flipCode=1)
    elif params['rotate'] == 180:
        sampledImage = cv2.flip(sampledImage, flipCode=-1)
    elif params['rotate'] == 270:
        sampledImage = cv2.transpose(sampledImage)
        sampledImage = cv2.flip(sampledImage, flipCode=0)

    sampledImage = \
        cv2.GaussianBlur(sampledImage, (params['blur'], params['blur']), 0)

    ret, sampledImage = \
         cv2.threshold(sampledImage, params['threshold'], 255, cv2.THRESH_BINARY)

    sampledImage = cv2.GaussianBlur(sampledImage, (3, 3), 0)
    return sampledImage

## Display an image to the user.
def showImage(image, windowName):
    """ Display an image to the user.

    :Parameters:
      - `image` (cv2 image) - image to be shown
      - `windowName` (string) - title to be shown on the window

    :Returns:
      None
    """
    cv2.namedWindow(windowName)
    cv2.imshow(windowName, image)
    cv2.waitKey(0)
    cv2.destroyWindow(windowName)

## Convert the image to text.
def performOCR(image):
    """ Convert the image to text.

    :Parameters:
      - `image` (cv2 image) - image to process

    :Returns:
      Text found in the image

    :Return Type:
      string
    """
    global api
    print "Performing OCR"
    height, width = image.shape
    channel1 = 1
    iplimage = cv.CreateImageHeader((width, height), cv.IPL_DEPTH_8U, channel1)
    cv.SetData(iplimage, image.tostring(), image.dtype.itemsize * (width))
    tesseract.SetCvImage(iplimage, api)
    text = api.GetUTF8Text()
    conf = api.MeanTextConf()
    text = text.replace(' ', '')
    return text

## Write text to a database.
def storeText(imageID, text, params):
    """ Write text to a database.

    :Parameters:
      - `imageID` (string) - a hopefully unique ID of the processed image
      - `text` (string) - text extracted from the image
      - `params` (dict) - parameteres used to sample the image

    :Returns:
      None
    """
    global sql_conn, sql_cur
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', unicode(text, 'UTF-8'))
        if (c in string.ascii_letters) or (c in string.digits)
    ).upper()
    print "Storing text for image %s:\n%s" % (imageID, text)
    if sql_cur and sql_conn:
        sql_cur.execute((
            "REPLACE INTO graves "
            "(ref, text, threshold, rotate, blur) "
            "VALUES (\"%s\", \"%s\", %d, %d, %d)") % (
            imageID,
            text,
            params['threshold'],
            params['rotate'],
            params['blur'],
        ))
        sql_conn.commit()

def getKnownWords(text):
    global knownWords
    return {"weight": 1.0, "words": [{"weight": 1.0, "word": w} for w in knownWords if w in string.upper(text)]}

def shutdownTools():
    global sql_conn, sql_cur
    if sql_cur:
        sql_cur.close()
    if sql_conn:
        sql_conn.close()


# If run as a standalone script
if __name__ == "__main__":
    sys.exit(main(sys.argv))    # Call main function, then provide exit status
