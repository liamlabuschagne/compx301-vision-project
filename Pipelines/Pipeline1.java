import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Point;
import org.opencv.core.CvType;

class Pipeline1 {
    public static void applyCLAHE(Mat src, int tileGridWidth, double clipLimit) {
        if (src.channels() >= 3) {
            System.out.println("Applying CLAHE");
            System.out.println("Grid: " + tileGridWidth + "x" + tileGridWidth);
            System.out.println("Clip Limit: " + clipLimit);

            Mat channel = new Mat();
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2Lab); // Gets the RGB color image and converts to Lab.

            // Get the L channel
            Core.extractChannel(src, channel, 0);

            // Apply the CLAHE to the the L channel
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setTilesGridSize(new Size(tileGridWidth, tileGridWidth));
            clahe.setClipLimit(clipLimit);
            clahe.apply(channel, channel);

            // Merge the color panes back into Lab img
            Core.insertChannel(channel, src, 0);

            // Convert back to RGB
            Imgproc.cvtColor(src, src, Imgproc.COLOR_Lab2BGR);

            // Release temporary Mat from memory
            channel.release();
        }
    }

    public static void binarise(Mat src, int blocksize, double C) {
        System.out.println("Applying adaptive binary thresholding using MEAN_C method.");
        System.out.println("Block size: " + blocksize);
        System.out.println("C value: " + C);
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blocksize, C);
    }

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

    public static void cvtColorSpace(Mat src, int code) {
        System.out.println("Converting to colour space with code: " + code);
        Imgproc.cvtColor(src, src, code);
    }

    public static void medianBlur(Mat src, int ksize) {
        System.out.println("Applying median blur");
        System.out.println("Kernel Size: " + ksize);
        Imgproc.medianBlur(src, src, ksize);
    }

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

    public static void shrinkGrow(Mat src, int ksize, int iterations) {
        System.out.println("Applying Shrink/Grow Noise Removal");
        System.out.println("Kernel Size: " + ksize);
        System.out.println("Iterations: " + iterations);

        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
        Point anchor = new Point(-1, -1);

        Imgproc.erode(src, src, element, anchor, iterations);
        Imgproc.dilate(src, src, element, anchor, iterations);
    }

    public static void pipeline(Mat src) {
        // applyCLAHE(src, 8, 4); // Contrast enhancement
        Imgcodecs.imwrite("step1.jpg", src);
        medianBlur(src, 5); // Remove some noise
        Imgcodecs.imwrite("step2.jpg", src);
        // gaussian(src, 11, 5);
        // sharpen(src, 10); // Sharpen to define the veins more
        Imgcodecs.imwrite("step3.jpg", src);
        cvtColorSpace(src, 6); // Convert to grayscale
        Imgcodecs.imwrite("step4.jpg", src);
        // sobel(src, 1, -1); // Apply Scharr edge detection
        Imgcodecs.imwrite("step5.jpg", src);
        binarise(src, 11, 1.01); // Binarise to convert to black and white
        Imgcodecs.imwrite("step6.jpg", src);
        shrinkGrow(src, 3, 2);
    }

    public static void main(String args[]) {
        if (args.length != 2) {
            System.out.println("Usage: java Pipeline1 <input1.jpg> <input2.jpg>");
            return;
        }

        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Mat src1 = Imgcodecs.imread(args[0]);
        pipeline(src1);
        Imgcodecs.imwrite("output.jpg", src1);
    }
}
