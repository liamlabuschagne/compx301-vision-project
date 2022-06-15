import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
public class RetinalMatch {

    public static byte saturate(double val){
            int iVal = (int)Math.round(val);
            iVal = iVal > 255 ? 255: (iVal < 0 ? 0 :iVal);
            return (byte)iVal;
    }
    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src=Imgcodecs.imread("../../RIDB/IM000001_2.jpg");
        Mat dst= Mat.zeros(src.size(), src.type());
        
        double alpha = 2.2; //Modifies contrast
        int beta = 50; //Modifies Brightness

        byte[] srcData = new byte[(int) (src.total()*src.channels())];
        src.get(0, 0, srcData);
        byte[] dstData = new byte[(int) (src.total() * src.channels())];
        for(int y = 0; y < src.rows(); y++){
            for(int x = 0; x < src.cols(); x++){
                for(int c = 0; c < src.channels(); c++){
                    double pixelVal = srcData[(y * src.cols() + x) * src.channels() + c];
                    pixelVal = pixelVal < 0 ? pixelVal + 256 : pixelVal;
                    dstData[(y * src.cols() + x) * src.channels() + c] = saturate(alpha * pixelVal + beta);
                }
            }
        }
        dst.put(0, 0, dstData);
        Imgcodecs.imwrite("Image_1_Contrast.jpg",dst);
        HighGui.imshow("Original", src);
        HighGui.imshow("Modified", dst);

        HighGui.waitKey();
        System.exit(0);
    }
}