import java.util.Arrays;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.ORB;

class RetinalMatch {
    public static void applyCLAHE(Mat src, int tileGridWidth, double clipLimit) {
        if (src.channels() >= 3) {
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
        Imgproc.adaptiveThreshold(src, src, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, blocksize, C);
    }

    public static void gaussian(Mat src, int ksize, double sigma) {
        Imgproc.GaussianBlur(src, src, new Size(ksize, ksize), sigma);
    }

    public static void sobel(Mat src, int order, int ksize) {
        // Apply Sobel in both directions
        Mat grad_x = new Mat(), grad_y = new Mat();
        Imgproc.Sobel(src, grad_x, -1, order, 0, ksize);
        Imgproc.Sobel(src, grad_y, -1, 0, order, ksize);

        // Add them together
        Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, src);
    }

    public static void laplace(Mat src, int ksize) {
        Imgproc.Laplacian(src, src, -1, ksize);
    }

    public static void cvtColorSpace(Mat src, int code) {
        Imgproc.cvtColor(src, src, code);
    }

    public static void medianBlur(Mat src, int ksize) {
        Imgproc.medianBlur(src, src, ksize);
    }

    public static void sharpen(Mat src, double amount) {
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
        Point anchor = new Point(-1, -1);

        Imgproc.dilate(src, src, new Mat(), anchor, iterations);
        Imgproc.erode(src, src, new Mat(), anchor, iterations);
    }

    public static Mat[] subset(Mat img) {
        int cols = img.cols();
        int rows = img.rows();

        Mat[] subsetImage = new Mat[4];

        // divide into 4 rectangular regions
        int middleRow = (int) Math.floor(rows / 2.0);
        int middleColumn = (int) Math.floor(cols / 2.0);
        subsetImage[0] = img.submat(0, middleRow, 0, middleColumn); // Top left corner
        subsetImage[1] = img.submat(middleRow, rows, 0, middleColumn); // Top right corner
        subsetImage[2] = img.submat(0, middleRow, middleColumn, cols); // bottom left corner
        subsetImage[3] = img.submat(middleRow, rows, middleColumn, cols); // bottom right corner

        return subsetImage;
    }

    public static double histEval(Mat src1, Mat src2) {

        // Setup parameters for calcHist
        int hBins = 50, sBins = 60;
        int[] histSize = { hBins, sBins };
        float[] ranges = { 0, 180, 0, 256 }; // hue varies from 0 to 179, saturation from 0 to 255
        int[] channels = { 0, 1 }; // Use the 0-th and 1-st channels

        // Calculate and normalise histograms
        Mat hist1 = new Mat(), hist2 = new Mat();
        List<Mat> hsv1List = Arrays.asList(src1);
        Imgproc.calcHist(hsv1List, new MatOfInt(channels), new Mat(), hist1, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist1, hist1, 0, 1, Core.NORM_MINMAX);

        List<Mat> hsv2List = Arrays.asList(src2);
        Imgproc.calcHist(hsv2List, new MatOfInt(channels), new Mat(), hist2, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist2, hist2, 0, 1, Core.NORM_MINMAX);

        return Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
    }

    public static int stepNumber = 1;

    public static void step(Mat src) {
        Imgcodecs.imwrite("step" + stepNumber + ".jpg", src);
        stepNumber++;
    }

    public static void pipeline(Mat src) {
        applyCLAHE(src, 8, 4); // Adaptive contrast enhancement
        sharpen(src, 10);
        gaussian(src, 11, 2000000);
        cvtColorSpace(src, 6);
        binarise(src, 11, 1.001);
        medianBlur(src, 15);
        shrinkGrow(src, 3, 2);
    }

    public static boolean isSame(String image1, String image2) {

        Mat src1 = Imgcodecs.imread(image1);
        Mat src2 = Imgcodecs.imread(image2);
        pipeline(src1);
        pipeline(src2);

        // -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
        ORB detector = ORB.create();
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint(), keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(src1, new Mat(), keypoints1, descriptors1);
        detector.detectAndCompute(src2, new Mat(), keypoints2, descriptors2);

        int similarity = 0;

        if (descriptors1.cols() == descriptors2.cols()) {
            MatOfDMatch matchMatrix = new MatOfDMatch();
            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            matcher.match(descriptors1, descriptors2, matchMatrix);
            DMatch[] matches = matchMatrix.toArray();

            for (DMatch match : matches)
                if (match.distance <= 50)
                    similarity++;
        }

        return similarity > 400;
    }

    public static void main(String args[]) {
        if (args.length != 2) {
            System.out.println("Usage: java RetinalMatch <input1.jpg> <input2.jpg>");
            return;
        }

        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        System.out.println(isSame(args[0], args[1]) ? "1" : "0");
    }
}
