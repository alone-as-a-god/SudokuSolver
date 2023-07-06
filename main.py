import imageConverter as ic
import cv2
from pyfiglet import Figlet
from PIL import ImageGrab


if __name__ == "__main__":
    f = Figlet(font='slant')
    print(f.renderText('Sudoku Solver'))
    input_image = input("Please enter the path to the image you want to solve (or leave empty to use current clipboard): ")  # replace with actual input

    # Tries to grab an image from the clipboard if no path is given
    if input_image == "":
        input_image = ImageGrab.grabclipboard()
        if input_image is None:
            print("No image found in clipboard")
            exit()
        input_image.save("clipboard.png")
        input_image = "clipboard.png"

    result_image = ic.handle_input(cv2.imread(input_image))
    if result_image is None:
        print("No board found")
    else:
        cv2.imshow("Input", result_image)
        cv2.waitKey(0)

