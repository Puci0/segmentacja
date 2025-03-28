import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg')
#img = cv2.imread('word-segmentation.JPEG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h,w,c = img.shape

if w > 1000:
        new_w = 1000
        ar = w/h
        new_h = int(new_w/ar)
        img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)

# plt.imshow(img)
# plt.show()

def binarisation(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

thresh_img = binarisation(img)
plt.show()

def dilatation(img,depth):
        kernel = np.ones((3,depth), np.uint8)
        eroded = cv2.erode(img, np.ones((2, 2), np.uint8), iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        return dilated

# plt.imshow(dilatation(thresh_img), cmap='gray')
# plt.show()

# ----------------- line segmentation -------------------------------
dilatedImg = dilatation(thresh_img,85)

(contours, heirarchy) = cv2.findContours(dilatedImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sorted_contours_lines = sorted(contours, key = lambda ctr : cv2.boundingRect(ctr)[1]) # (x, y, w, h)

img2 = img.copy()

for ctr in sorted_contours_lines:
        x, y, w, h = cv2.boundingRect(ctr)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (40, 100, 250), 2)

# plt.imshow(img2)
# plt.show()

# ------------------------- word segmentation ---------------------------------

dilatedImg2 = dilatation(thresh_img,20)

img3 = img.copy()
words_list = []

for line in sorted_contours_lines:

    # roi of each line
    x, y, w, h = cv2.boundingRect(line)
    roi_line = dilatedImg2[y:y + h, x:x + w]

    # draw contours on each word
    (cnt, heirarchy) = cv2.findContours(roi_line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contour_words = sorted(cnt, key=lambda cntr: cv2.boundingRect(cntr)[0])

    for word in sorted_contour_words:

        if cv2.contourArea(word) < 400:
            continue

        x2, y2, w2, h2 = cv2.boundingRect(word)
        words_list.append([x + x2, y + y2, x + x2 + w2, y + y2 + h2])
        cv2.rectangle(img3, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (0, 255, 0), 2)

# plt.imshow(img3)
# plt.show()
# ----------------------------- letter segmentation -----------------------------------
img_letters = img.copy()
letters_list = []

for word in words_list:
    x1, y1, x2, y2 = word
    roi_word = img[y1:y2, x1:x2]
    thresh_word = binarisation(roi_word)
    contours, _ = cv2.findContours(thresh_word.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours_letters = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    for letter in sorted_contours_letters:
        if cv2.contourArea(letter) < 5:
            continue

        x_l, y_l, w_l, h_l = cv2.boundingRect(letter)
        letters_list.append([x1 + x_l, y1 + y_l, x1 + x_l + w_l, y1 + y_l + h_l])
        cv2.rectangle(img_letters, (x1 + x_l, y1 + y_l), (x1 + x_l + w_l, y1 + y_l + h_l), (255, 0, 0), 2)

plt.imshow(img_letters)
plt.show()
#
#index = 2
# for index in letters_list:
#     x1, y1, x2, y2 = index
#     letter_img = img[y1:y2, x1:x2]
#
#     plt.imshow(letter_img, cmap='gray')
#     plt.show()