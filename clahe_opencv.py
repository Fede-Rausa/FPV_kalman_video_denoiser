import cv2



class clahe_free:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8), mode='hsv'):
        '''
        Initialize the CLAHE (Contrast Limited Adaptive Histogram Equalization) object.
        clipLImit: argument of cv2.createCLAHE. Default 2.0.
        tileGridSize: argument of cv2.createCLAHE. Default (8, 8).
        mode: The color space mode to be used ('rgb', 'gray', 'hsv', 'lab').
        set to 'gray' for grayscale images. default is 'hsv'
        '''

        class_select = {'hsv': clahe_hsv, 'lab': clahe_lab,
                        'rgb': clahe_rgb, 'gray': clahe_gray}

        newclahe = class_select[mode](clipLimit, tileGridSize)
        
        self.clahe = newclahe.clahe
        self.apply = newclahe.apply




class clahe_rgb:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def apply(self, frame):
        b, g, r = cv2.split(frame)
        b_clahe = self.clahe.apply(b)
        g_clahe = self.clahe.apply(g)
        r_clahe = self.clahe.apply(r)
        return cv2.merge([b_clahe, g_clahe, r_clahe])



class clahe_gray:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def apply(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)
    


class clahe_hsv:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_clahe = self.clahe.apply(v)
        # Merge back and convert to BGR
        frame_hsv_clahe = cv2.merge([h, s, v_clahe])
        frame_clahe = cv2.cvtColor(frame_hsv_clahe, cv2.COLOR_HSV2BGR)
        return frame_clahe



class clahe_lab:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def apply(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        frame_lab_clahe = cv2.merge([l_clahe, a, b])
        frame_clahe = cv2.cvtColor(frame_lab_clahe, cv2.COLOR_LAB2BGR)
        return frame_clahe