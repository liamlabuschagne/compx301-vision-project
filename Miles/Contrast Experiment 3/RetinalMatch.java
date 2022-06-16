import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;
public class RetinalMatch {


    //CLAHE being contrast limited adpative histogram enhancement.
    //This algorithm adapts upon histogram equilization, limiting the contrast amplification by computing several histograms, each corresponding to sections of the image.
    //This reduces noise as it operates on smaller regions of the image
    //When applying CLAHE to a color image on the luminance channel we get better equilization for all channels of the bgr image as the lightness is redistributed evenly.
    public static void applyCLAHE(Mat src, Mat dst){

        if(src.channels() >= 3){
            Mat channel = new Mat();
            Imgproc.cvtColor(src, dst, Imgproc.COLOR_BGR2Lab); //Gets the RGB color image and converts to Lab.

            //Get the L channel
            Core.extractChannel(dst, channel, 0);

            //Apply the CLAHE to the the L channel
            CLAHE clahe = Imgproc.createCLAHE();
            clahe.setClipLimit(4);
            clahe.apply(channel, channel);

            //Merge the color panes back into Lab img
            Core.insertChannel(channel, dst, 0);

            //Convert back to RGB
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_Lab2BGR);

            //Release temporary Mat from memory
            channel.release();
        }
    }

    public static void main(String[] args){
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src=Imgcodecs.imread("../../RIDB/IM000001_2.jpg");
        Mat dst = src.clone();
        //Mat dst= new Mat();
        //List<Mat> channels = new LinkedList();
        //Core.split(src, channels);
        //CLAHE clahe = Imgproc.createCLAHE();
        //Mat dst = new Mat(src.getHeight(),src.getWidth(), CvType.CV_8UC4);
        //clahe.apply(channels.get(0), dst);
        //Core.merge(channels, src);
        applyCLAHE(src, dst);
        Imgcodecs.imwrite("Image_1_Contrast_2.jpg",dst);
        HighGui.imshow("Original", src);
        HighGui.imshow("Modified", dst);

        HighGui.waitKey();
        System.exit(0);
    }
}