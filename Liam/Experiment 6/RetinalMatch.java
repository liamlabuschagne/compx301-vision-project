import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

class RetinalMatch {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Thresholding");
            System.out.println("Usage: java RetinalMatch <input.jpg> <output.jpg> <threshold 0-255>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        System.out.println("Loading file " + args[0]);
        Mat src = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_GRAYSCALE);

        // Creating an empty matrices to store the destination image.
        Mat dst = new Mat(src.rows(), src.cols(), src.type());

        // Applying simple threshold
        double threshold = Double.parseDouble(args[2]);
        double thresholdUsed = Imgproc.threshold(src, dst, threshold, 255, Imgproc.THRESH_BINARY);
        System.out.println("Applying threshold of " + thresholdUsed);
        System.out.println("Outputting to file " + args[1]);
        Imgcodecs.imwrite(args[1], dst);
    }
}
