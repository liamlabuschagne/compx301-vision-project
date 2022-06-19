import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.Collectors;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
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
import org.opencv.core.DMatch;
import org.opencv.features2d.SIFT;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.xfeatures2d.SURF;

class Pipeline2 {
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
        double hessianThreshold = 400;
        int nOctaves = 1;// 4;
        int nOctaveLayers = 1;// 3;
        boolean extended = false, upright = false;
        SURF detector = SURF.create(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
        MatOfKeyPoint keypoints1 = new MatOfKeyPoint(), keypoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat(), descriptors2 = new Mat();
        detector.detectAndCompute(src1, new Mat(), keypoints1, descriptors1);
        detector.detectAndCompute(src2, new Mat(), keypoints2, descriptors2);
        // -- Step 2: Matching descriptor vectors with a FLANN based matcher
        // Since SURF is a floating-point descriptor NORM_L2 is used
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);
        // -- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.7f;
        List<DMatch> listOfGoodMatches = new ArrayList<>();
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodMatches = new MatOfDMatch();
        goodMatches.fromList(listOfGoodMatches);

        // -- Draw matches
        Mat imgMatches = new Mat();
        Features2d.drawMatches(src1, keypoints1, src2, keypoints2, goodMatches, imgMatches, Scalar.all(-1),
                Scalar.all(-1), new MatOfByte(), 2);

        // Imgcodecs.imwrite("output1.jpg", src1);
        // Imgcodecs.imwrite("output2.jpg", src2);
        // Imgcodecs.imwrite("matches.jpg", imgMatches);

        System.out.println("Good Matches: " + goodMatches.size(0));
        return goodMatches.size(0) > 190;
    }

    public static void main(String args[]) {
        if (args.length != 2) {
            System.out.println("Usage: java Pipeline2 <input1.jpg> <input2.jpg>");
            return;
        }

        // Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        int tests = 100;

        String group1[] = new String[tests];
        String group2[] = new String[tests];

        for (int i = 0; i < tests; i++) {
            int imageN = (int) (Math.random() * 5) + 1;
            int personN = (int) (Math.random() * 20) + 1;
            String filename = "../RIDB/IM00000" + imageN + "_" + personN + ".jpg";
            group1[i] = filename;
        }

        for (int i = 0; i < tests; i++) {
            int imageN = (int) (Math.random() * 5) + 1;
            int personN = (int) (Math.random() * 20) + 1;
            String filename = "../RIDB/IM00000" + imageN + "_" + personN + ".jpg";
            group2[i] = filename;
        }

        int correct = 0;
        for (int i = 0; i < tests; i++) {
            boolean same = isSame(group1[i], group2[i]);
            int person1 = Integer.parseInt(group1[i].substring(17, group1[i].length() -
                    4));
            int person2 = Integer.parseInt(group2[i].substring(17, group2[i].length() -
                    4));
            boolean samePerson = person1 == person2;
            System.err.println("Comparing " + group1[i] + " and " + group2[i]);
            System.err.println("Same: " + (same ? "Yes" : "No"));
            if (same == samePerson) {
                correct++;
            } else {
                System.out.println("Incorrect!");
            }
            System.err.println(correct + "/" + tests);
        }

        // System.out.println("Same: " + (isSame(args[0], args[1]) ? "Yes" : "No"));
    }
}
