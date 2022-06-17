import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import java.util.stream.Collectors;

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
import org.opencv.dnn.KeypointsModel;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.features2d.SIFT;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;

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

    // public static void cvtBinToGray(Mat src, int code) {
    // System.out.println("Converting to colour space with code: " + code);
    // Imgproc.cvtColor(src, src, Imgproc.CV_GRAY);
    // }

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

        Imgproc.dilate(src, src, new Mat(), anchor, iterations);
        Imgproc.erode(src, src, new Mat(), anchor, iterations);

        // Imgproc.erode(src, src, element, anchor, iterations);
        // Imgproc.dilate(src, src, element, anchor, iterations);
    }

    public static Mat[] subset(Mat img) {
        int cols = img.cols();
        int rows = img.rows();

        Mat[] subsetImage = new Mat[4];
        List<Mat> subsetList = Arrays.asList(subsetImage);

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
        // step(src);
        sharpen(src, 10);
        // step(src);
        gaussian(src, 11, 2000000);
        // step(src);
        // cvtColorSpace(src, 6);
        // step(src);
        // binarise(src, 11, 1.001);
        // // step(src);
        medianBlur(src, 15);
        // // step(src);
        shrinkGrow(src, 3, 2);
        step(src);
        cvtColorSpace(src, Imgproc.COLOR_BGR2HSV);
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
        Mat src2 = Imgcodecs.imread(args[1]);
        pipeline(src2);

        Mat[] parts1 = subset(src1);
        Mat[] parts2 = subset(src2);

        double similarity = 0;

        for (int i = 0; i < 4; i++) {
            similarity += 0.25 * histEval(parts1[i], parts2[i]);
        }

        boolean samePerson = args[0].charAt(9) == args[1].charAt(9);

        System.out.println("Similarity: " + similarity);
        System.out.println("Same: " + (similarity > 0.98 ? "Yes" : "No"));
        System.out.println("Correct: " + (similarity > 0.98 && samePerson ? "Yes" : "No"));

        Imgcodecs.imwrite("output1.jpg", src1);
        Imgcodecs.imwrite("output2.jpg", src2);
    }
}
