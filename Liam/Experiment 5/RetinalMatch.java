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
            System.out.println("Histogram Comparison");
            System.out.println("Usage: java RetinalMatch <input1.jpg> <input2.jpg>");
            System.out.println("Retruns: A decimal value describing the similarity of the histograms of these images.");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the images.
        Mat src1 = Imgcodecs.imread(args[0]);
        Mat src2 = Imgcodecs.imread(args[1]);
        if (src1.empty() || src2.empty()) {
            System.err.println("Cannot read the images");
            System.exit(0);
        }

        // Convert to HSV
        Mat hsv1 = new Mat(), hsv2 = new Mat();
        Imgproc.cvtColor(src1, hsv1, Imgproc.COLOR_BGR2HSV);
        Imgproc.cvtColor(src2, hsv2, Imgproc.COLOR_BGR2HSV);

        // Setup parameters for calcHist
        int hBins = 50, sBins = 60;
        int[] histSize = { hBins, sBins };
        float[] ranges = { 0, 180, 0, 256 }; // hue varies from 0 to 179, saturation from 0 to 255
        int[] channels = { 0, 1 }; // Use the 0-th and 1-st channels

        // Calculate and normalise histograms
        Mat hist1 = new Mat(), hist2 = new Mat();
        List<Mat> hsv1List = Arrays.asList(hsv1);
        Imgproc.calcHist(hsv1List, new MatOfInt(channels), new Mat(), hist1, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist1, hist1, 0, 1, Core.NORM_MINMAX);

        List<Mat> hsv2List = Arrays.asList(hsv2);
        Imgproc.calcHist(hsv2List, new MatOfInt(channels), new Mat(), hist2, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist2, hist2, 0, 1, Core.NORM_MINMAX);

        // Comare using all four methods.
        for (int compareMethod = 0; compareMethod < 4; compareMethod++) {
            double similarity = Imgproc.compareHist(hist1, hist2, compareMethod);
            System.out.println("Similarity using method " + compareMethod + " was " + similarity);
        }
    }
}
