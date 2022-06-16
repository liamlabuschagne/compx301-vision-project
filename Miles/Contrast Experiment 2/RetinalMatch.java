import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
public class RetinalMatch {

    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src=Imgcodecs.imread("../../RIDB/IM000001_2.jpg");
        
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat dst= new Mat();
        Imgproc.equalizeHist(src, dst);

        Imgcodecs.imwrite("Image_1_Contrast_2.jpg",dst);
        HighGui.imshow("Original", src);
        HighGui.imshow("Modified", dst);

        HighGui.waitKey();
        System.exit(0);
    }
}