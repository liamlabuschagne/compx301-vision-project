import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class Sharpening {

    public static void sharpen(Mat src, double amount) {
        System.out.println("Sharpening");
        System.out.println("Amount: " + amount);

        Mat centreOne = Mat.zeros(3, 3, CvType.CV_32F);
        centreOne.put(1, 1, 1);

        Mat cross = Mat.zeros(3, 3, CvType.CV_32F);
        cross.put(0, 1, 1);
        cross.put(1, 0, 1);
        cross.put(1, 1, 1);
        cross.put(2, 1, 1);
        cross.put(1, 2, 1);

        // Divide cross by 5
        Core.divide(cross, new Scalar(5), cross);

        // Create empty Mat to hold the final kernel
        Mat kernel = new Mat(3, 3, CvType.CV_32F);

        // Subtract it from centreOne
        Core.subtract(centreOne, cross, kernel);

        // Multiply this result by amount
        Core.multiply(kernel, new Scalar(amount), kernel);

        // Add centreOne one more time
        Core.add(kernel, centreOne, kernel);

        // Convolve the kernel
        Imgproc.filter2D(src, src, -1, kernel);
    }

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Sharpening");
            System.out.println("Usage: java Sharpening <input.jpg> <output.jpg> <amount>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load the src image
        System.out.println("Loading image: " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);

        // Apply the filter
        double amount = Double.parseDouble(args[2]);
        sharpen(src, amount);

        // Save the dst image
        System.out.println("Saving image: " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
