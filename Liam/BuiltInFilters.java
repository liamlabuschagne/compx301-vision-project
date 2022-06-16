import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

class BuiltInFilters {

    public static void gaussian(Mat src, int ksize, double sigma) {
        System.out.println("Applying gaussian filter");
        System.out.println("Kernel Size: " + ksize);
        System.out.println("Sigma: " + sigma);
        Imgproc.GaussianBlur(src, src, new Size(ksize, ksize), sigma);
    }

    public static void sobel(Mat src, int order, int ksize) {
        System.out.println("Applying sobel filter");
        System.out.println("Derivative order: " + order);
        System.out.println("Kernel Size: " + ksize);
        // Apply Sobel in both directions
        Mat grad_x = new Mat(), grad_y = new Mat();
        Imgproc.Sobel(src, grad_x, -1, order, 0, ksize);
        Imgproc.Sobel(src, grad_y, -1, 0, order, ksize);

        // Add them together
        Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, src);
    }

    public static void laplace(Mat src, int ksize) {
        System.out.println("Applying laplace filter");
        System.out.println("Kernel Size: " + ksize);
        Imgproc.Laplacian(src, src, -1, ksize);
    }

    public static void showUsage() {
        System.out.println("Built-in filters");
        System.out.println("Usage: java BuiltInFilters <filter>");
        System.out.println("<filter> can be one of gaussian, laplace or sobel.");
    }

    public static void main(String[] args) {

        if (args.length == 0) {
            showUsage();
            return;
        }

        switch (args[0]) {
            case "gaussian":
                if (args.length != 5) {
                    System.out.println(
                            "Usage: java BuiltInFilters gaussian <input.jpg> <output.jpg> <ksize> <sigma>");
                    return;
                }
                break;
            case "laplace":
                if (args.length != 4) {
                    System.out.println("Usage: java BuiltInFilters laplace <input.jpg> <output.jpg> <ksize>");
                    return;
                }
                break;
            case "sobel":
                if (args.length != 5) {
                    System.out.println(
                            "Usage: java BuiltInFilters sobel <input.jpg> <output.jpg> <derivative order> <ksize>");
                    return;
                }
                break;
            default:
                showUsage();
                return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load in the source image
        System.out.println("Loading image: " + args[1]);
        Mat src = Imgcodecs.imread(args[1]);

        // Create an empty image with the same metadata
        Mat dst = new Mat(src.size(), src.type());

        switch (args[0]) {
            case "gaussian":
                int ksize = Integer.parseInt(args[3]);
                double sigma = Double.parseDouble(args[4]);
                dst = gaussian(src, ksize, sigma);
                break;
            case "sobel":
                int order = Integer.parseInt(args[3]);
                ksize = Integer.parseInt(args[4]);
                dst = sobel(src, order, ksize);
                break;
            case "laplace":
                ksize = Integer.parseInt(args[3]);
                dst = laplace(src, ksize);
                break;
        }
        // Apply the filter and output to dst image
        System.out.println("Saving image " + args[2]);
        Imgcodecs.imwrite(args[2], dst);
    }
}
