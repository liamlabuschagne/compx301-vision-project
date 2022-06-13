import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

class RetinalMatch {

    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Sharpening");
            System.out.println("Usage: java RetinalMatch <input.jpg> <output.jpg> <sharpening factor n>");
            return;
        }

        String inputFile = args[0];
        String outputFile = args[1];
        int sharpeningFactor = Integer.parseInt(args[2]);

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load in the source image
        System.out.println("Loading input file: " + inputFile);
        Mat src = Imgcodecs.imread(inputFile);

        // Create an empty image with the same metadata
        Mat dst = new Mat(src.size(), src.type());

        Mat centreOne = Mat.zeros(3, 3, CvType.CV_32F);
        centreOne.put(1, 1, 1);
        System.out.println("Centre One:");
        System.out.println(centreOne.dump());
        System.out.println();

        Mat cross = Mat.zeros(3, 3, CvType.CV_32F);
        cross.put(0, 1, 1);
        cross.put(1, 0, 1);
        cross.put(1, 1, 1);
        cross.put(2, 1, 1);
        cross.put(1, 2, 1);

        // Divide cross by 5
        Core.divide(cross, new Scalar(5), cross);
        System.out.println("Cross:");
        System.out.println(cross.dump());
        System.out.println();

        Mat kernel = new Mat(3, 3, CvType.CV_32F);

        // Subtract it from centreOne
        Core.subtract(centreOne, cross, kernel);
        System.out.println("centreOne - cross:");
        System.out.println(kernel.dump());
        System.out.println();

        // Multiply this result by sharpeningFactor
        Core.multiply(kernel, new Scalar(sharpeningFactor), kernel);
        System.out.println("Mulitiply by scaling factor:");
        System.out.println(kernel.dump());
        System.out.println();

        // Add centreOne one more time
        Core.add(kernel, centreOne, kernel);
        System.out.println("Add centreOne again:");
        System.out.println(kernel.dump());
        System.out.println();

        Imgproc.filter2D(src, dst, -1, kernel);

        // Apply the filter and output to dst image
        System.out.println("Outputting to " + outputFile);
        Imgcodecs.imwrite(outputFile, dst);
    }
}
