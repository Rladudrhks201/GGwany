import cv2


def image_show(image, windowName='show', close=False):
    cv2.imshow(windowName, image)
    cv2.waitKey(0)
    if close:
        cv2.destroyAllWindows()


