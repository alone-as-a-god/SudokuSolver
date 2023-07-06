import imageConverter as ic
import cv2

if __name__ == "__main__":
    print("Sudoku Solver") # make this fancy later
    input_image = "bug.png" # replace with actual input
    cv2.imshow("Input", ic.handle_input(cv2.imread(input_image)))
    cv2.waitKey(0)

