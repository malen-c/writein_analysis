import pytesseract
import cv2
from PIL import Image
import numpy as np
import re
from os import listdir

#This needs to be a path to the Tesseract engine on your local machine
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
template_folder = './templates/'

candidate = 'TeressaRaiford'

style_regex = re.compile('[^C0-9]')
writein_regex = re.compile('[^a-zA-Z]')
config = '--psm 7 --oem 3'

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
h_mending_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,1))

orb = cv2.ORB_create()
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#There are many ballot styles, so this dict has 2-tuples of the minimum y and x values of the write-in line region for each style that had 1+ write-ins.
#A third field codes for whether there is whitespace to the right of the write-in line, because many people continue into this space if it's present.
#Additionally, the included 4th field has the filenames of the ballots which the 2-tuples were captured from for feature based image alignment.

ballot_styles = {
    35: (2310, 890, 0, 'AB-0001+10109.jpg'), 21: (2166, 890, 1, 'AB-0002+10185.jpg'), 33: (2310, 890, 1, 'AB-0002+10191.jpg'),
    46: (2310, 880, 1, 'AB-0002+10269.jpg'), 24: (2164, 890, 1, 'AB-0002+10329.jpg'), 15: (2164, 890, 1, 'AB-0002+10341.jpg'),
    36: (2360, 897, 0, 'AB-0002+10361.jpg'), 9 : (2159, 894, 0, 'AB-0002+10403.jpg'), 19: (2362, 895, 1, 'AB-0003+10041.jpg'),
    17: (2358, 895, 1, 'AB-0003+10097.jpg'), 83: (2210, 890, 1, 'AB-0003+10269.jpg'), 41: (2318, 895, 0, 'AB-0004+10109.jpg'),
    48: (2210, 890, 1, 'AB-0004+10379.jpg'), 62: (2170, 890, 1, 'AB-0006+10081.jpg'), 31: (2362, 885, 1, 'AB-0006+10353.jpg'),
    44: (2320, 890, 1, 'AB-0008+10181.jpg'), 47: (2315, 890, 1, 'AB-0008+10273.jpg'), 63: (2162, 890, 1, 'AB-0008+10647.jpg'),
    1 : (610, 1283, 0, 'AB-0008+10667.jpg'), 37: (2366, 890, 0, 'AB-0009+10239.jpg'), 4 : (2177, 890, 0, 'AB-0012+10243.jpg'),
    43: (2323, 886, 1, 'AB-0013+10003.jpg'), 25: (2172, 898, 1, 'AB-0016+10787.jpg'), 16: (2158, 890, 0, 'AB-0017+10529.jpg'),
    13: (2156, 900, 1, 'AB-0018+10089.jpg'), 55: (2212, 895, 0, 'AB-0019+10733.jpg'), 50: (2211, 892, 0, 'AB-0020+10123.jpg'),
    42: (2306, 892, 0, 'AB-0023+10377.jpg'), 58: (2167, 897, 1, 'AB-0028+10507.jpg'), 57: (2167, 898, 1, 'AB-0034+10619.jpg'),
    49: (2218, 888, 0, 'AB-0066+10283.jpg'), 14: (2375, 897, 1, 'AB-0077+10083.jpg'), 82: (616, 1288, 0, 'AB-0086+10267.jpg'),
    30: (2168, 895, 1, 'AB-0283+10443.jpg'), 2 : (2174, 887, 1, 'AB-0321+10619.jpg'), 86: (2218, 884, 0, 'AB-0423+10311.jpg'),
    10: (2158, 895, 0, 'AB-0622+10777.jpg'), 65: (625, 1294, 0, 'AB-0754+10499.jpg'), 40: (2358, 895, 1, 'AB-0917+10595.jpg')
}


# Puts image into OpenCV format, does some filtering that is helpful for all applications
def clean_ballot(image_path):

    i = cv2.imread(image_path, 0)
    ib = cv2.GaussianBlur(i, (5,5), 0) #pre-filtering
    ret,ibo = cv2.threshold(i,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #Otsu thresholding to clean up image
    
    return ibo

# Helper function for get_border_points, fits lines corresponding to the border of the voting area then finds intersections, returning 3
# for the calculation of the affine transformation
def intersect_finder(l_border, top_border, r_border, bot_border):

    #fitting a line to each of the borders of the image. l_m, l_b represent slope and intercept of line

    l_x = l_border[:,0]
    l_y = l_border[:,1]
    l_m, l_b = np.polyfit(l_x[l_x!=0], l_y[l_y!=0], deg = 1)

    t_x = top_border[:,0]
    t_y = top_border[:,1]
    t_m, t_b = np.polyfit(t_x[t_x!=0], t_y[t_y!=0], deg = 1)

    r_x = r_border[:,0]
    r_y = r_border[:,1]
    r_m, r_b = np.polyfit(r_x[r_x!=0], r_y[r_y!=0], deg = 1)
    
    b_x = bot_border[:,0]
    b_y = bot_border[:,1]
    b_m, b_b = np.polyfit(b_x[b_x!=0], b_y[b_y!=0], deg = 1)


    #calculating intersects and rounding to whole numbers for pixels
    
    tl_point_x = (l_b - t_b)/(t_m - l_m)
    tl_point   = [round(i) for i in [(tl_point_x * l_m) + l_b, tl_point_x]]
    
    tr_point_x = (r_b - t_b)/(t_m - r_m)
    tr_point   = [round(i) for i in [(tr_point_x * r_m) + r_b, tr_point_x]]

    br_point_x = (r_b - b_b)/(b_m - r_m)
    br_point   = [round(i) for i in [(br_point_x * r_m) + r_b, br_point_x]]

    bl_point_x = (l_b - b_b)/(b_m - l_m)
    bl_point   = [round(i) for i in [(bl_point_x * l_m) + l_b, bl_point_x]]

    return [tl_point, tr_point, br_point, bl_point]


# Returns the (y, x) values of intersections of border lines
def get_border_points(cv_image): #takes an image, returns an array of in order the top left, top right, bottom right, bottom
                                 #left corners of ballot area
    
    cv_image_x, cv_image_y = len(cv_image[0]), len(cv_image)
    
    #masking all of the ballot stuff so it just aligns based on the rectangles
    cv_image_border = cv_image.copy()
    cv_image_border = cv2.rectangle(cv_image_border,
                                   (int(cv_image_x * .037), int(cv_image_y * .037)),
                                   (int(cv_image_x * .961), int(cv_image_y * .957)),
                                   (255,255,255),-1)

    ret, thresh = cv2.threshold(cv_image_border, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    box_contours = []
    
    for cont in contours:

        arr_cont = np.concatenate(cont, axis = 0) #friendlier format

        x = np.take(arr_cont, [0], axis = 1) #array of x values
        y = np.take(arr_cont, [1], axis = 1) #array of y values

        x_range = np.ptp(x)
        y_range = np.ptp(y)

        if x_range in range(25, 75) and y_range in range(0, 30):
            box_contours.append(cont)

    top_border, r_border, bot_border, l_border = np.zeros((500, 2), int), np.zeros((500, 2), int), np.zeros((500, 2), int), np.zeros((500, 2), int)

    easy_box_contours = np.concatenate(np.concatenate(box_contours, axis = 0), axis = 0) #changes box_contours from an array of arrays of arrays to just an
                                                                                         #array of arrays. my life is a joke

    x_min, x_max = min(np.take(easy_box_contours, [0], axis = 1)), max(np.take(easy_box_contours, [0], axis = 1))
    y_min, y_max = min(np.take(easy_box_contours, [1], axis = 1)), max(np.take(easy_box_contours, [1], axis = 1))


    #this loop separates the boxes into the top, right, left, and bottom strips of boxes, and also grabs the most extreme points of each
    #in order to draw a line of best fit corresponding to the border of the ballot area

    t_ind, r_ind, b_ind, l_ind = 0, 0, 0, 0
    
    for cont in box_contours:

        arr_cont = np.concatenate(cont, axis = 0)
        
        x = np.take(arr_cont, [0], axis = 1)
        x_range = max(x) - min(x) + 1
        y = np.take(arr_cont, [1], axis = 1)
        y_range = max(y) - min(y) + 1
        
        if min(y) - 30 < y_min:
            for point in cont:
                #in order, point is in top third of contour, point is not in the rightmost 2.5%, not in the leftmost 2.5%
                if point[0][1] < min(y) + y_range*.33 and point[0][0] < max(x) - x_range*.025 and point[0][0] > min(x) + x_range*.025:
                    top_border[t_ind] = point
                    t_ind += 1

        if max(x) + 30 > x_max:
            for point in cont:
                #point is in right 15% of contour, point not in bottom 10%, not in top 10%
                if point[0][0] > max(x) - x_range*.15 and point[0][1] < max(y) - y_range*.1 and point[0][1] > min(y) + y_range*.1:
                    r_border[r_ind] = point
                    r_ind += 1

        if max(y) + 30 > y_max:
            for point in cont:
                #point is in bottom third, not in right 2.5%, not in left 2.5%
                if point[0][1] > max(y) - y_range*.33 and point[0][0] < max(x) - x_range*.025 and point[0][0] > min(x) + x_range*.025:
                    bot_border[b_ind] = point
                    b_ind += 1

        if min(x) - 30 < x_min:
            for point in cont:
                #point is in left 15%, not in bottom 10%, not in top 10%
                if point[0][0] < min(x) + x_range*.15 and point[0][1] < max(y) - y_range*.1 and point[0][1] > min(y) + y_range*.1:
                    l_border[l_ind] = point
                    l_ind += 1

    points = intersect_finder(l_border, top_border, r_border, bot_border)
    
    return np.float32(points)


# Processes image, crops to write_in box based on minimum y and x values 
def extract_vote(cv_image, y_min, x_min, whitespace):
    
    y_max = y_min + 52
    x_max = x_min + 355 + (whitespace * 135)
    
    inverted_img = cv2.bitwise_not(cv_image)
    writein_line = cv2.morphologyEx(inverted_img, cv2.MORPH_OPEN, horizontal_kernel, iterations = 2) #extracting the line that they're supposed
                                                                                                     #to write in on
    h_contours = cv2.findContours(writein_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_contours = h_contours[0] if len(h_contours) == 2 else h_contours[1] #removing the line
    for c in h_contours:
        cv2.drawContours(cv_image, [c], -1, (255,255,255), 2)

    cv_image = 255 - cv2.morphologyEx(255 - cv_image, cv2.MORPH_CLOSE, h_mending_kernel, iterations=1)  #adding back in some handwriting that gets cut when
                                                                                                        #the line's removed
    potential_votes = []

    for i in range(-5, 6):
        img = Image.fromarray(cv_image[y_min + 5*i : y_max + 5*i, x_min : x_max])
        unfiltered_t = pytesseract.image_to_string(img, config = config) 
        potential_votes.append(writein_regex.sub('', unfiltered_t)) #removes all characters that pytesseract interprets that aren't letters or spaces

    return potential_votes
    
# Calculates the Levenshtein distance of two strings. x = write-in, y = target string
def l_distance(x, y = candidate):   

    x = x.lower()
    y = y.lower()

    if x == '':
        return len(y)

    while x[0] == ' ':
        x = x[1:]
    
    x_len = len(x) + 1
    y_len = len(y) + 1
    
    mat = np.zeros((x_len, y_len))
    for i in range(x_len):
        mat[i,0] = i
    for j in range(y_len):
        mat[0,j] = j

    for i in range(1, x_len):
        for j in range(1, y_len):
            if min(i, j) == 0:
                mat[i,j] = max(i,j)
            else:
                if x[i-1] == y[j-1]:
                    mat[i,j] = min(mat[i-1, j]  + 1,
                                   mat[i, j-1]  + 1,
                                   mat[i-1,j-1]
                                )
                else:
                    mat[i,j] = min(mat[i-1, j]  + 1,
                                   mat[i, j-1]  + 1,
                                   mat[i-1,j-1] + 1
                                )
    return int(mat[x_len - 1, y_len - 1])

# Does everything! Takes an image name(i.e. 'AB-0196+10369.jpg') and style ID and returns best guess from pytesseract, array of Levenshtein distances
def read_ballot(b_path, style_id):

    cv_image = clean_ballot(b_path)
    y_min, x_min, whitespace, template = ballot_styles[style_id]

    im_points = get_border_points(cv_image)
    te_points = template_points[template]

    M = cv2.getPerspectiveTransform(im_points, te_points)
    warp_cv_image = cv2.warpPerspective(cv_image, M, (cv_image.shape[1], cv_image.shape[0]))

    potential_votes = extract_vote(warp_cv_image, y_min, x_min, whitespace)
    best_guess = min(potential_votes, key = l_distance)

    
    return best_guess, min([l_distance(vote, candidate) for vote in potential_votes])

# Grabbing corner points of template images for transform

templates = listdir(template_folder)
template_points = {}

for template in templates:
    cv_template = clean_ballot(template_folder + template)
    te_points = get_border_points(cv_template)
    template_points[template] = te_points
