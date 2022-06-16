import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;
import org.opencv.core.Core;
import org.opencv.core.Size;

class ShrinkGrow {

    public static void shrinkGrow(Mat src, int ksize, int iterations) {
        System.out.println("Applying Shrink/Grow Noise Removal");
        System.out.println("Kernel Size: " + ksize);
        System.out.println("Iterations: " + iterations);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
        Point anchor = new Point(-1, -1);

        Imgproc.erode(src, src, element, anchor, iterations);
        Imgproc.dilate(src, src, element, anchor, iterations);
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.out.println("Shrink-Grow Noise Removal");
            System.out.println("Usage: java ShrinkGrow <input.jpg> <output.jpg> <ksize> <iterations>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        System.out.println("Loading image: " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);
        int ksize = Integer.parseInt(args[2]);
        int iterations = Integer.parseInt(args[3]);
        shrinkGrow(src, ksize, iterations);

        // Output image
        System.out.println("Saving image: " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
