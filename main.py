import imageConverter as ic
import cv2

if __name__ == "__main__":
    print("Sudoku Solver") # make this fancy later
    input = "sample.png" # replace with actual input
    cv2.imshow("Input", ic.handle_input(input))
    cv2.waitKey(0)


