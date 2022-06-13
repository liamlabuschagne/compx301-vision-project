import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Core;

class RetinalMatch {

    public static void main(String[] args) {

        if (args.length != 2) {
            System.out.println("Grayscale Conversion");
            System.out.println("Usage: java RetinalMatch <input.jpg> <output.jpg>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Read input file as grayscale
        System.out.println("Reading input file as grayscale " + args[0]);
        Mat src = Imgcodecs.imread(args[0], Imgcodecs.IMREAD_GRAYSCALE);

        // Output grayscale version as file
        System.out.println("Outputting file " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
