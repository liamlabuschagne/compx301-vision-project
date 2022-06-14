import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;

class RetinalMatch {

    public static void main(String[] args) {
        if (args.length < 4 || (args.length < 6 && !args[0].equals("laplace"))) {
            System.out.println("Built-in filters");
            System.out.println("Options are:");
            System.out.println("Sobel: java RetinalMatch sobel <input.jpg> <output.jpg> <dx> <dy> <ksize>");
            System.out
                    .println("Gaussian: java RetinalMatch gaussian <input.jpg> <output.jpg> <ksize> <sigmaX> <sigmaY>");
            System.out
                    .println("Laplace: java RetinalMatch laplace <input.jpg> <output.jpg> <ksize>");
            return;
        }

        // load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // Load in the source image
        System.out.println("Loading input file: " + args[1]);
        Mat src = Imgcodecs.imread(args[1]);

        // Create an empty image with the same metadata
        Mat dst = new Mat(src.size(), src.type());

        System.out.println("Applying " + args[0] + " filter.");
        switch (args[0]) {
            case "guassian":
                int ksize = Integer.parseInt(args[3]);
                double sigmaX = Double.parseDouble(args[4]);
                double sigmaY = Double.parseDouble(args[5]);
                Imgproc.GaussianBlur(src, dst, new Size(ksize, ksize), sigmaX, sigmaY);
                break;
            case "sobel":
                // Apply sobel in both directions
                Imgproc.Sobel(src, dst, -1, 1, 0,
                        Integer.parseInt(args[5]));
                Mat dst2 = new Mat(src.size(), src.type());
                Imgproc.Sobel(src, dst2, -1, 0, 1,
                        Integer.parseInt(args[5]));
                // Then add the results
                Core.add(dst, dst2, dst);
                break;
            case "laplace":
                ksize = Integer.parseInt(args[3]);
                Imgproc.Laplacian(src, dst, -1, ksize);
                break;
        }
        // Apply the filter and output to dst image
        System.out.println("Outputting to " + args[2]);
        Imgcodecs.imwrite(args[2], dst);
    }
}
