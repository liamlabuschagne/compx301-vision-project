import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Binarise {

    public static Mat binarise(Mat src, int blocksize, double C) {
        // Creating an empty matrices to store the destination image.
        Mat dst = new Mat(src.rows(), src.cols(), src.type());
        System.out.println("Applying adaptive thresholding.");
        Imgproc.adaptiveThreshold(src, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blocksize, C);
        return dst;
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
        System.out.println("Loading file " + args[0]);
        Mat src = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_GRAYSCALE);

        Mat dst = binarise(src, Integer.parseInt(args[2]), Double.parseDouble(args[3]));

        System.out.println("Outputting to file " + args[1]);
        Imgcodecs.imwrite(args[1], dst);
    }
}
