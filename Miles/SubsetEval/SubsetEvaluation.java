import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfFloat;
import org.opencv.core.Core;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
import java.util.List;
import java.util.Arrays;

public class SubsetEvaluation {

    public static Mat[] subset(Mat img, int grid){
        int cols = img.cols();
        int rows = img.rows();
        
        Mat[] subsetImage = new Mat[4];
        List<Mat> subsetList = Arrays.asList(subsetImage);

            //divide into 4 rectangular regions
            int middleRow = (int) Math.floor(rows / 2.0);
            int middleColumn = (int) Math.floor(cols / 2.0);
            subsetImage[0] = img.submat(0, middleRow, 0, middleColumn); //Top left corner
            subsetImage[1] = img.submat(middleRow, rows, 0, middleColumn); //Top right corner
            subsetImage[2] = img.submat(0, middleRow, middleColumn, cols); //bottom left corner
            subsetImage[3] = img.submat(middleRow, rows, middleColumn, cols); //bottom right corner
        // subsetList = Arrays.asList(subsetImage);
        // Imgproc.calcHist(subsetImage[2], new MatOfInt(channels), new Mat(), hist3, new MatOfInt(histSize),
        // new MatOfFloat(ranges), false);
        // Core.normalize(hist3, hist3, 0, 1, Core.NORM_MINMAX);

        // subsetList = Arrays.asList(subsetImage);
        // Imgproc.calcHist(subsetImage[3], new MatOfInt(channels), new Mat(), hist4, new MatOfInt(histSize),
        // new MatOfFloat(ranges), false);
        // Core.normalize(hist4, hist4, 0, 1, Core.NORM_MINMAX);
        return subsetImage;
    }
    public static void histEval(Mat hist1, Mat hist2){
        int hBins = 50, sBins = 60;
        int[] histSize = {hBins, sBins};
        float[] ranges = { 0, 180, 0, 256 }; // hue varies from 0 to 179, saturation from 0 to 255
        int[] channels = { 0, 1 }; // Use the 0-th and 1-st channels
        List<Mat> hsv1List = Arrays.asList(hist1);
        Imgproc.calcHist(hsv1List, new MatOfInt(channels), new Mat(), hist1, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist1, hist1, 0, 1, Core.NORM_MINMAX);

        List<Mat> hsv2List = Arrays.asList(hist2);
        Imgproc.calcHist(hsv2List, new MatOfInt(channels), new Mat(), hist2, new MatOfInt(histSize),
                new MatOfFloat(ranges), false);
        Core.normalize(hist2, hist2, 0, 1, Core.NORM_MINMAX);
        for (int compareMethod = 0; compareMethod < 4; compareMethod+=3) {
            double similarity = Imgproc.compareHist(hist1, hist2, compareMethod);
            String betterText = compareMethod % 2 == 0 ? "(higher is more similar)" : "(lower is more similar)";
            System.out.println("Similarity using method " + compareMethod + " was " + similarity + " " + betterText);   //Method 0 and Method 3 give numbers below 1. Method 0 is correlation, and Method 3 is Bhattacharyya.
            
        }
    }
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat img1 = Imgcodecs.imread("../../RIDB/IM000001_1.jpg");
        Mat img2 = Imgcodecs.imread("../../RIDB/IM000001_2.jpg");
        //Mat img2 = Imgcodecs.imread("../../RIDB/IM000001_2.jpg");
        Mat[] subImage = subset(img1, 4);
        Mat[] subImage2 = subset(img2, 4);
        histEval(subImage[0], subImage2[0]);
        histEval(subImage[1], subImage2[1]);
        histEval(subImage[2], subImage2[2]);
        histEval(subImage[3], subImage2[3]);
    }
}
