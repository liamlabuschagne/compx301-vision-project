import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

class ColorSpace {

    public static Mat cvtColorSpace(Mat src, int code) {
        System.out.println("Converting to colour space with code: " + code);
        Imgproc.cvtColor(src, src, code);
        return src;
    }

    public static void main(String[] args) {

        if (args.length != 3) {
            System.out.println("Colourspace Conversion");
            System.out.println("Usage: java ColorSpace <input.jpg> <output.jpg> <colour space code>");
            System.out.println("where colour space code is one of the OpenCV BGR2_ codes.");
            System.out.println("Useful codes:");
            System.out.println("GRAY: 6");
            System.out.println("HLS: 52");
            System.out.println("HLS FULL: 68");
            System.out.println("HSV: 40");
            System.out.println("HSV FULL: 66");
            System.out.println("XYZ: 32");
            System.out.println("YCrCb: 36");
            System.out.println("Lab: 44");
            System.out.println("Luv: 50");
            System.out.println("Yuv: 82");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Read input file
        System.out.println("Reading input file  " + args[0]);
        Mat src = Imgcodecs.imread(args[0]);

        // Convert to new colourspace
        int code = Integer.parseInt(args[2]);
        src = cvtColorSpace(src, code);

        // Output converted file
        System.out.println("Outputting file " + args[1]);
        Imgcodecs.imwrite(args[1], src);
    }
}
