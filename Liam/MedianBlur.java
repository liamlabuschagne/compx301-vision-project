import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class MedianBlur {

    public static void medianBlur(Mat src, int ksize) {
        System.out.println("Applying median blur");
        System.out.println("Kernel Size: " + ksize);
        Imgproc.medianBlur(src, src, ksize);
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Median Blur");
            System.out.println("Usage: java MedianBlur <input.jpg> <output.jpg> <ksize>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the image
        System.out.println("Loading image: " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);

        // Apply median blur
        int ksize = Integer.parseInt(args[2]);
        medianBlur(src, ksize);

        // Output image
        System.out.println("Saving image: " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
