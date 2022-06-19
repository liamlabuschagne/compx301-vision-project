import org.opencv.core.Core;
import org.opencv.core.Core.MinMaxLocResult;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class TemplateMatch {

    public static void binarise(Mat src, int blocksize, double C) {
        System.out.println("Applying adaptive binary thresholding using MEAN_C method.");
        System.out.println("Block size: " + blocksize);
        System.out.println("C value: " + C);
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blocksize, C);
    }

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src = null;
        // Load image file
        src = Imgcodecs.imread("../../RIDB/IM000001_1.jpg");
        Mat template = Imgcodecs.imread("binarisedImage.jpg");

        // binarise(template, 11, 1.001);

        Mat outputImage = new Mat();
        int matchMethod = Imgproc.TM_CCOEFF;
        // Template matching method
        Imgproc.matchTemplate(src, template, outputImage, matchMethod);

        MinMaxLocResult mmr = Core.minMaxLoc(outputImage);
        Point matchLoc = mmr.maxLoc;
        // Draw rectangle on result image
        Imgproc.rectangle(src, matchLoc, new Point(matchLoc.x + template.cols(),
                matchLoc.y + template.rows()), new Scalar(255, 255, 255));

        Imgcodecs.imwrite("templateMatch.jpg", src);
        System.out.println("Finished Template Match.");
    }
}
