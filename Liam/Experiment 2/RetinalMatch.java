import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

class RetinalMatch {

    public static void main(String[] args) {

        if (args.length < 3) {
            System.out.println(
                    "Please provide two file names, one input one output.\nThird parameter is the filter type, options are: box, gaussian and laplace.");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load in the source image
        System.out.println("Loading input file: " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);

        // Create an empty image with the same metadata
        Mat dst = new Mat(src.size(), src.type());

        // Prepare the filter
        Mat kernel = new Mat(3, 3, CvType.CV_32F);
        switch (args[2]) {
            case "gaussian":
                // Compass directions are 5
                kernel.setTo(new Scalar(5.0 / 40.0));
                // Corners are 3
                kernel.put(0, 0, 3.0 / 40.0);
                kernel.put(2, 0, 3.0 / 40.0);
                kernel.put(0, 2, 3.0 / 40.0);
                kernel.put(2, 2, 3.0 / 40.0);
                // Centre is 8
                kernel.put(1, 1, 8.0 / 40.0);
                break;
            case "laplace":
                kernel = new Mat(5, 5, CvType.CV_32F);
                kernel.setTo(new Scalar(0)); // Start off with 0's

                // Diagonal 1's
                kernel.put(2, 0, -1);
                kernel.put(1, 1, -1);
                kernel.put(0, 2, -1);
                kernel.put(1, 3, -1);
                kernel.put(2, 4, -1);
                kernel.put(3, 3, -1);
                kernel.put(4, 2, -1);
                kernel.put(3, 1, -1);

                // Compass direction -2's
                kernel.put(1, 2, -2);
                kernel.put(3, 2, -2);
                kernel.put(2, 1, -2);
                kernel.put(2, 3, -2);

                // Centre is 16
                kernel.put(2, 2, 16);
                break;
            case "box":
            default:
                kernel.setTo(new Scalar(1.0 / 9.0));
                break;
        }

        System.out.println("Applying " + args[2] + " filter with following matrix:");
        System.out.println(kernel.dump());

        // Apply the filter and output to dst image
        System.out.println("Outputting to " + args[1]);
        Imgproc.filter2D(src, dst, -1, kernel);
        Imgcodecs.imwrite(args[1], dst);
    }
}
