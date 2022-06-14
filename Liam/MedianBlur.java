import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class MedianBlur {

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Median Blur");
            System.out.println("Usage: java MedianBlur <input.jpg> <output.jpg> <ksize>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        System.out.println("Loading file " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);

        // Creating an empty matrix to store the destination image.
        Mat dst = new Mat(src.rows(), src.cols(), src.type());

        // Apply median blur
        int ksize = Integer.parseInt(args[2]);
        System.out.println("Applying median blur with " + ksize + " ksize");
        Imgproc.medianBlur(src, dst, ksize);

        // Output image
        System.out.println("Outputting file " + args[1]);
        Imgcodecs.imwrite(args[1], dst);
    }
}
