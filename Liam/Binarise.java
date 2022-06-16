import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Binarise {

    public static void binarise(Mat src, int blocksize, double C) {
        System.out.println("Applying adaptive binary thresholding using MEAN_C method.");
        System.out.println("Block size: " + blocksize);
        System.out.println("C value: " + C);
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blocksize, C);
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.out.println("Thresholding");
            System.out.println("Usage: java Binarise <input.jpg> <output.jpg> <blocksize> <C>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        System.out.println("Loading image " + args[0]);
        Mat src = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_GRAYSCALE);

        binarise(src, Integer.parseInt(args[2]), Double.parseDouble(args[3]));

        System.out.println("Saving image " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
