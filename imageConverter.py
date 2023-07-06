import cv2
import numpy
import tensorflow
import imutils
import solver

input_size = 48


# Function to split the board into its 81 individual cells
def split_board(board):
    rows = numpy.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = numpy.hsplit(r, 9)
        for box in cols:  # Resize each cell to 48x48 pixels and append to boxes
            box = cv2.resize(box,
                             (input_size, input_size)) / 255.0  # Last part is to normalize the data (from 0-255 to 0-1)
            boxes.append(box)

    return boxes


# Function to get the perspective of the sudoku board
def get_perspective(img, location, width=900, height=900):
    pts1 = numpy.float32([location[0], location[3], location[1], location[2]])
    pts2 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


# Used to put the image back into the original perspective
# Takes the original image as input
def get_inv_perspective(img, masked_num, location, height=900, width=900):
    pts1 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = numpy.float32([location[0], location[3], location[1], location[2]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result


# Function to locate a Sudoku board within an image
def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)  # Apply bilateral filter to remove noise
    edged = cv2.Canny(bfilter, 30, 180)  # Apply Canny Edge Detection
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]  # Sort contours by area and keep largest 15

    location = None
    for contour in contours:  # Find the biggest contour with 4 corners (the Sudoku board)
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break

    if location is None:  # If no Sudoku board is found
        return None, None

    result = get_perspective(img, location)  # Get the perspective of the Sudoku board
    return result, location  # Return the board and its location


def display_numbers(img, numbers, color=(0, 255, 255)):
    w = int(img.shape[1] / 9)  # Width of each cell
    h = int(img.shape[0] / 9)  # Height of each cell
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:  # If the number is not zero then display
                cv2.putText(img, str(numbers[(j * 9) + i]), (i * w + int(w / 2) - int((w / 4)), int((j + 0.7) * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2,
                            cv2.LINE_AA)  # Places number into its corresponding cell

    return img


def handle_input(image):
    board, location = find_board(image)  # Find the Sudoku board
    if board is None or location is None:  # if no board is found return the original image
        return image

    # Converts the board into a 9x9 array of grayscale cells
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    cells = split_board(gray)
    cells = numpy.array(cells).reshape(-1, input_size, input_size, 1)

    classes = numpy.arange(0, 10)  # Creates an array consisting of numbers from 0-9
    model = tensorflow.keras.models.load_model("model.h5")
    predictions = model.predict(cells)  # Uses an OCR model to get possible numbers for each cell
    predicted_numbers = []

    for i in predictions:
        index = (numpy.argmax(i))  # Finds the index of the highest probability
        predicted_number = classes[index]  # Gets the number corresponding to the index
        predicted_numbers.append(predicted_number)  # Appends the number to the list

    read_board = numpy.array(predicted_numbers).reshape(9, 9)  # Reshapes the list into a 9x9 array

    solver.solve_sudoku(read_board, 0, 0)  # Solves the Sudoku board

    # Creates a mask array where 0 is a cell that was originally empty and 1 is a cell that was originally filled
    bin_arr = numpy.where(numpy.array(predicted_numbers) > 0, 0, 1)

    # Collapses read_board into one dimension and then multiplies the solved board by the mask array
    # That way we're left with the numbers that were added by the solving function
    flat_solved_board_nums = read_board.flatten() * bin_arr

    mask = numpy.zeros_like(board)  # Creates a zero array with the same shape as the board
    res = display_numbers(mask, flat_solved_board_nums)  # Displays the solved cells onto the mask array
    inv = get_inv_perspective(image, res, location)  # Applies the perspective of the original picture to the solved board

    combined = cv2.addWeighted(image, 0.7, inv, 1, 0)  # Combines the original image with the solved board
    return combined
