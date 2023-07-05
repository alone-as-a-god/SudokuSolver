import cv2
import numpy
import tensorflow
import imutils
import solver

input_size = 48


def split_board(board):
    rows = numpy.vsplit(board, 9)
    i = 0
    boxes = []
    for r in rows:
        cols = numpy.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0
            boxes.append(box)

    return boxes


def get_perspective(img, location, width=900, height=900):
    pts1 = numpy.float32([location[0], location[3], location[1], location[2]])
    pts2 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


def find_board(image):
    img = cv2.imread(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break

    result = get_perspective(img, location)
    return result, location

def display_numbers(img, numbers, color=(0, 255, 9)):
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)

    return img

def handle_input(image):
    board, location = find_board(image)
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    rois = split_board(gray)
    rois = numpy.array(rois).reshape(-1, input_size, input_size, 1)


    classes = numpy.arange(0, 10)
    model = tensorflow.keras.models.load_model("model-OCR.h5")
    predictions = model.predict(rois)
    predicted_numbers = []

    for i in predictions:
        index = (numpy.argmax(i))
        predicted_number = classes[index]
        predicted_numbers.append(predicted_number)

    read_board = numpy.array(predicted_numbers).reshape(9, 9)

    solver.solve_sudoku(read_board, 0, 0)


    binArr = numpy.where(numpy.array(predicted_numbers)>0, 0, 1)

    flat_solved_board_nums = read_board.flatten()*binArr

    mask = numpy.zeros_like(board)
    res = display_numbers(mask, flat_solved_board_nums)
    combined = cv2.addWeighted(board, 0.5, res, 0.5, 0)
    return combined
